import cv2
import numpy as np
import polars as pl
from PIL import Image as PIL_Image


def odd(n: int) -> int:
    """Гарантирует, что n — положительное нечётное число."""
    n = max(3, int(n))
    return n if n % 2 == 1 else n + 1

def slice_df_to_crop(df: pl.DataFrame, x1: int, y1: int, x2: int, y2: int) -> pl.DataFrame:
    """Отфильтровать слова, чей центр внутри (x1,y1,x2,y2), и пересчитать координаты
    так, чтобы (x1,y1) стало (0,0). Даёт img2table работать с кропом без
    перезапуска Tesseract."""
    if df.is_empty():
        return df
    cx = (pl.col("x1") + pl.col("x2")) / 2
    cy = (pl.col("y1") + pl.col("y2")) / 2
    filtered = df.filter((cx >= x1) & (cx <= x2) & (cy >= y1) & (cy <= y2))
    return filtered.with_columns([
        (pl.col("x1") - x1).alias("x1"), (pl.col("x2") - x1).alias("x2"),
        (pl.col("y1") - y1).alias("y1"), (pl.col("y2") - y1).alias("y2"), ]
    )


class ImageProcessor:

    def __init__(self, target_dpi: int, image: PIL_Image.Image, no_grayscale: bool = False):
        if target_dpi is None or target_dpi <= 0:
            raise ValueError(f"target_dpi must be a positive integer, got {target_dpi}")
        self.target_dpi = target_dpi

        if not no_grayscale:
            self.image, self.gray_array = self.keep_only_pure_grayscale(image, tolerance=45)
        else:
            self.image, self.gray_array = image, np.asarray(image)

        self.MIN_LINE_LEN: int = int(1.2 * self.target_dpi)
        self.BLOCK_SIZE: int = odd( max( 3, int(0.25 * self.target_dpi) ) )
        self.CLIPPED_TOLERANCE_RATIO: float = 0.75   # доля длины ROI, которую должна покрывать линия
        self.LINE_GAP_CLOSE_RATIO: float = 0.15      # закрываем разрывы вдоль линии до этой доли DPI
        self.EDGE_TOLERANCE_PX: int = max(5, int(0.05 * self.target_dpi))
        self.DEDUP_IOU: float = 0.9                  # два бокса с IoU выше — считаются одним и тем же
        self.MAX_DEPTH: int = 5
        self.PAD_PX: int = 3
        self.SHRINK: int = max(self.PAD_PX * 2, round(0.06 * self.target_dpi))
        self._candidates: list[dict] = []

    @staticmethod
    def keep_only_pure_grayscale(pil_img: PIL_Image.Image, tolerance: int = 45, light_threshold: int = 200) -> tuple[PIL_Image.Image, np.ndarray]:
        """Убирает цветные пиксели (штампы, водяные знаки, подсветки) и слишком светлые."""
        np_img = np.asarray(pil_img)

        if np_img.ndim == 2:
            gray = np_img
            color_mask = np.zeros_like(gray, dtype=bool)
        else:
            gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
            max_ch = np_img.max(axis=2).astype(np.int16)
            min_ch = np_img.min(axis=2).astype(np.int16)
            color_mask = (max_ch - min_ch) > int(1.5 * tolerance)

        light_mask = gray > light_threshold

        h, w = gray.shape
        small = cv2.resize(gray, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA)
        bg = int(np.median(small))

        result = np.full_like(gray, bg)
        keep_mask = ~(color_mask | light_mask)
        np.copyto(result, gray, where=keep_mask)

        return PIL_Image.fromarray(result, mode='L'), result


    def _count_full_span_lines(self, lines: np.ndarray, area: tuple[int, int, int, int], axis: int) -> int:
        """Считает линии, которые реально тянутся через ROI. axis=1 — горизонтальные, axis=0 — вертикальные.
        Перед подсчётом закрываем разрывы ВДОЛЬ линии (выцветшие чернила на сканах) через morph CLOSE.
        Дальше считаем связные компоненты: наклонная при сканировании линия лежит диагонально,
        но в 8-связности она остаётся одним компонентом — его ширина/высота и есть её реальная длина.
        Линия считается полноразмерной, если её длина >= CLIPPED_TOLERANCE_RATIO от длины ROI."""
        x1, y1, x2, y2 = area
        length = (x2 - x1) if axis == 1 else (y2 - y1)
        if length <= 2 * self.EDGE_TOLERANCE_PX:
            return 0

        roi = lines[y1:y2, x1:x2]
        if roi.size == 0:
            return 0

        gap_close = max(3, int(self.LINE_GAP_CLOSE_RATIO * self.target_dpi))
        if axis == 1:
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (gap_close, 1))
        else:
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, gap_close))
        roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, k)

        num_labels, _, stats, _ = cv2.connectedComponentsWithStats((roi > 0).astype(np.uint8), connectivity=8)
        threshold = max(1, int(length * self.CLIPPED_TOLERANCE_RATIO))

        count = 0
        for i in range(1, num_labels):  # 0 — фон, пропускаем
            line_length = stats[i, cv2.CC_STAT_WIDTH] if axis == 1 else stats[i, cv2.CC_STAT_HEIGHT]
            if line_length >= threshold:
                count += 1
        return count

    def _find_row_bands(self, hor_lines: np.ndarray, img_h: int, img_w: int) -> list[tuple[int, int, int, int]]:
        """Делит изображение на горизонтальные полосы по разрывам в hor_lines.
        Каждая полоса гарантированно содержит хотя бы одну горизонтальную линию.
        Короткие разрывы (< 0.25 DPI) не считаются границей — это пробелы внутри таблицы."""
        min_gap = int(0.25 * self.target_dpi)
        has_line = hor_lines.sum(axis=1) > 0  # True в строках, где есть линия

        # Морфологически закрываем короткие разрывы (< min_gap), чтобы не резать внутри таблицы
        has_line = np.convolve(has_line.astype(np.uint8),
                               np.ones(min_gap, dtype=np.uint8), mode='same') > 0

        bands, in_band, start_y = [], False, 0
        for y, active in enumerate(has_line):
            if active and not in_band:
                start_y, in_band = y, True
            elif not active and in_band:
                bands.append((0, max(0, start_y - self.PAD_PX),
                              img_w, min(img_h, y + self.PAD_PX)))
                in_band = False
        if in_band:
            bands.append((0, max(0, start_y - self.PAD_PX), img_w, img_h))

        return bands if len(bands) > 1 else [(0, 0, img_w, img_h)]

    def _collect_candidates(self, grid: np.ndarray, hor_lines: np.ndarray, ver_lines: np.ndarray,
                            region: tuple[int, int, int, int], depth: int = 0) -> None:
        """Рекурсивно собирает все плотные сетки в self._candidates. На каждом уровне уменьшает
        дилатацию — это разлепляет вложенные структуры: рамку и таблицу внутри;
        две таблицы, стоящие в одном листе через текстовый разрыв. Решение 'рамка это или таблица'
        принимается позже, в _filter_frame_wrappers — сюда просто сгребаем всех возможных."""
        rx1, ry1, rx2, ry2 = region
        region_w, region_h = rx2 - rx1, ry2 - ry1
        min_w, min_h = round(2.0 * self.target_dpi), round(0.6 * self.target_dpi)
        if region_w < min_w or region_h < min_h:
            return

        dil_scale = 0.15 * (0.5 ** depth)
        dil_k = max(3, round(dil_scale * self.target_dpi))
        region_grid = grid[ry1:ry2, rx1:rx2]
        grid_dilated = cv2.dilate( region_grid,
                                   cv2.getStructuringElement(cv2.MORPH_RECT, (dil_k, dil_k)), iterations=2 )
        contours, _ = cv2.findContours( grid_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )

        img_h, img_w = grid.shape

        for cnt in contours:
            crx, cry, crw, crh = cv2.boundingRect(cnt)
            if crw < min_w or crh < min_h:
                continue

            abs_x1, abs_y1 = rx1 + crx, ry1 + cry
            abs_x2, abs_y2 = abs_x1 + crw, abs_y1 + crh

            fx1, fy1 = max(0, abs_x1 - self.PAD_PX), max(0, abs_y1 - self.PAD_PX)
            fx2, fy2 = min(img_w, abs_x2 + self.PAD_PX), min(img_h, abs_y2 + self.PAD_PX)
            bbox = (fx1, fy1, fx2, fy2)

            n_h = self._count_full_span_lines(hor_lines, bbox, axis=1)
            n_v = self._count_full_span_lines(ver_lines, bbox, axis=0)

            if n_h >= 3 or n_v >= 3:
                self._candidates.append( {"bbox": bbox, "n_h": n_h, "n_v": n_v, "depth": depth} )

            if depth < self.MAX_DEPTH:
                sub = ( int(abs_x1 + self.SHRINK), int(abs_y1 + self.SHRINK),
                        int(abs_x2 - self.SHRINK), int(abs_y2 - self.SHRINK) )
                if sub[2] - sub[0] >= min_w and sub[3] - sub[1] >= min_h:
                    self._collect_candidates( grid, hor_lines, ver_lines, sub, depth=depth + 1 )


    def _filter_frame_wrappers(self) -> list[dict]:
        """Сначала дедуплицируем: один и тот же контур может найтись на нескольких глубинах
        через разные ветки рекурсии. Затем отбрасываем обёртки — боксы, внутри которых лежит
        настоящий меньший кандидат (не просто его shrink-тень)."""

        def is_drilldown_shadow(outer: dict, inner: dict) -> bool:
            """inner выглядит как последовательный drill-down shrink outer по SHRINK на каждом уровне."""
            d = inner["depth"] - outer["depth"]
            if d <= 0:
                return False
            expected_shrink = d * self.SHRINK
            ob, ib = outer["bbox"], inner["bbox"]
            exp = ( ob[0] + expected_shrink, ob[1] + expected_shrink,
                    ob[2] - expected_shrink, ob[3] - expected_shrink )
            tol = max(self.PAD_PX * 2, self.SHRINK // 2)
            return all( abs(exp[i] - ib[i]) <= tol for i in range(4) )

        def iou(b1: tuple[int, int, int, int], b2: tuple[int, int, int, int]) -> float:
            ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
            ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
            if ix2 <= ix1 or iy2 <= iy1:
                return 0.0
            inter = (ix2 - ix1) * (iy2 - iy1)
            a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
            a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
            return inter / (a1 + a2 - inter)
        def is_contained(inner_b: tuple, outer_b: tuple, tol: int) -> bool:
            return (inner_b[0] >= outer_b[0] - tol and
                    inner_b[1] >= outer_b[1] - tol and
                    inner_b[2] <= outer_b[2] + tol and
                    inner_b[3] <= outer_b[3] + tol)

        # 1. Дедуп. Кандидат — дубликат уже добавленного, если IoU > DEDUP_IOU (тот же контур в другой ветке)
        #    ИЛИ он является shrink-тенью уже добавленного. Оставляем того, у кого меньше depth.
        unique: list[dict] = []
        for c in sorted(self._candidates, key=lambda x: x["depth"]):
            duplicate = False
            for u in unique:
                if iou(c["bbox"], u["bbox"]) > self.DEDUP_IOU or is_drilldown_shadow(u, c):
                    duplicate = True
                    break
            if not duplicate:
                unique.append(c)

        # 2. Убираем обёртки: бокс считается обёрткой, если внутри него лежит другой кандидат,
        #    существенно меньший по площади и не являющийся его drilldown-тенью.
        contain_tol = max(self.PAD_PX * 4, self.SHRINK)
        answer = []
        for c in unique:
            cb = c["bbox"]
            c_area = (cb[2] - cb[0]) * (cb[3] - cb[1])
            is_wrapper = False
            for other in unique:
                if other is c:
                    continue
                ob = other["bbox"]
                o_area = (ob[2] - ob[0]) * (ob[3] - ob[1])
                if (is_contained(ob, cb, tol=contain_tol) and
                        o_area < c_area * 0.85 and
                        not is_drilldown_shadow(c, other)):
                    is_wrapper = True
                    break
            if not is_wrapper:
                answer.append(c)
        return answer

    def get_table_bboxes(self) -> tuple[list[tuple[int, int, int, int]], np.ndarray]:
        img_h, img_w = self.gray_array.shape

        thresh = cv2.adaptiveThreshold( self.gray_array, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                        cv2.THRESH_BINARY_INV, self.BLOCK_SIZE, 8 )
        hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.MIN_LINE_LEN, 1))
        ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, self.MIN_LINE_LEN))
        hor_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, hor_kernel)
        ver_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, ver_kernel)
        grid = cv2.add(hor_lines, ver_lines)

        bands = self._find_row_bands(hor_lines, img_h, img_w)
        for band in bands:
            self._collect_candidates(grid, hor_lines, ver_lines, band, depth=0)
        tables = self._filter_frame_wrappers()
        return [t["bbox"] for t in tables], grid


    @staticmethod
    def draw_table_borders(crop_pil: PIL_Image.Image, grid_full: np.ndarray, bbox_full: tuple[int, int, int, int],
                           line_thickness: int = 2):
        """Дорисовывает внешнюю рамку таблицы для img2table.extract_tables. Выделено в функцию на случай,
        если придётся дорисовывать не только внешние, но и внутренние линии (используя grid_full + bbox_full)."""
        arr = np.array(crop_pil)
        if arr.ndim == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        h, w = arr.shape[:2]
        cv2.rectangle(arr, (0, 0), (w - 1, h - 1), color=0, thickness=line_thickness)
        return PIL_Image.fromarray(arr, mode='L')
