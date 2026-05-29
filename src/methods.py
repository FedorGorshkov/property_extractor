# Built-in modules
from typing import *
import re
from datetime import datetime

# External modules
from img2table.tables.objects.extraction import ExtractedTable

# Project modules
from src import global_variables as gv
from src import funcs

def func_props(confidence: float = None, group: int = None):
    def decorator(func):
        if confidence:
            func.confidence = confidence
        if group:
            func.group = group
        return func
    return decorator

class Solver:
    def __init__(self, tag, task, possible_values, possible_uoms, condition: str = None, alt_keywords: List[str] = None, flag: int = None, desc_for_llm: str = None):
        self.tag: str = tag
        self.task: dict = task
        self.possible_values: List[str] = possible_values
        self.possible_uoms: List[str] = possible_uoms
        self.condition: str = condition
        self.alt_keywords: List[str] = alt_keywords
        self.flag: int = flag
        self.desc_for_llm: str = desc_for_llm
        self.result, self.uom, self.confidence = None, None, 0.0
        self.methods = [func for func in Solver.__dict__.values() if hasattr(func, "confidence")]
        self.methods.sort(key=lambda x: x.confidence, reverse=True)

        self.pure_text: str = None
        self.passport: ExtractedTable = None
        self.tech_specs_frame: str = None
        self.tables: List[ExtractedTable] = None
        self.object_type = ""

        self.now_executing: Callable = None

    def set_file(self, file_dict: dict):
        self.pure_text, self.tables = file_dict["pure_text"], file_dict["tables"]
        if "tech_specs" in file_dict:
            self.tech_specs_frame = file_dict["tech_specs"]
        if "passport" in file_dict:
            self.passport = file_dict["passport"]
        if "object_type" in file_dict:
            self.object_type = file_dict["object_type"]

    def solve(self):
        # TODO: more complicated consensus search
        if self.condition == "search_in_header":
            self.try_find_in_header()
            if self.result:
                if not (self.result.startswith(gv.LLM_FAILURE_FLAG) or self.result.endswith(gv.LLM_FAILURE_FLAG)):
                    return
        for func in self.methods:
            triada = (self.result, self.uom, func.confidence)
            self.now_executing = func.__name__
            func(self)
            if not isinstance(self.result, str):
                self.result, self.uom, self.confidence = triada
                continue
            if gv.validate_result(self.result) or func.confidence < triada[2]:
                self.result, self.uom, self.confidence = triada
                continue
            if self.possible_values:
                if self.result not in self.possible_values:
                    self.clarify_ans()
                if not gv.validate_result(self.result):
                    self.result, self.uom, self.confidence = triada
                    continue
            if gv.validate_result(self.result):
                self.confidence = func.confidence
                break
        if isinstance(self.alt_keywords, list):
            if (self.task["en_name"] not in self.alt_keywords) and (self.task["ru_name"] not in self.alt_keywords):
                for alt_keyword in self.alt_keywords:
                    if gv.has_russian_letters(alt_keyword):
                        self.task["ru_name"] = alt_keyword
                    else:
                        self.task["en_name"] = alt_keyword
                    self.solve()

        if not gv.validate_result(self.result) and self.tech_specs_frame:
            self.search_in_tech_specs()
        if not gv.validate_result(self.result) and self.flag == 1:
            pass

    def try_find_in_header(self):
        if not self.passport:
            return
        point, ru = funcs.search_in_tables([self.passport], self.task["ru_name"]), True
        if not point:
            point, ru = funcs.search_in_tables([self.passport], self.task["en_name"]), False
        if point:
            point = point[0]
            self.execute_task(self.passport, ru, point, bypass_src=True)

    def execute_task(self, src: Union[str, List[ExtractedTable], ExtractedTable], take_ru_name: int, points: list[int], bypass_src: bool = False) -> Tuple[str, str]:
        if not isinstance(src, str) and not isinstance(src, list) and not isinstance(src, ExtractedTable):
            raise ValueError(f"src must be str or ExtractedTable, not '{type(src)}'")
        if isinstance(src, list):
            if not isinstance(src[0], ExtractedTable):
                raise ValueError(f"src must be list of ExtractedTable, not '{type(src[0])}'")
        if not points and not bypass_src:
            return
        constraints, prop_name = "", self.task["ru_name"] if take_ru_name == 1 else self.task["en_name"]
        if self.condition != "only_if_mentioned":
            prompt = f"""{('Ты извлекаешь свойства для объекта "' + self.object_type + '".') if self.object_type else ""}
            Найди значение свойства и его единицу измерения в переданном {"тексте" if isinstance(src, str) else f"{gv.CURRENT_TABLE_REPR} представлении таблицы"}
            из корпоративного документа. Ты должен отвечать в формате json с двумя полями: value и uom. Если значения свойства нет, заполняй value {gv.LLM_FAILURE_FLAG},
            если свойство не требует единицы измерения или она не найдена, так же, заполняй uom {gv.LLM_FAILURE_FLAG}. Пример твоего ответа: {{"value": "80", "uom": "дБА"}}.
             Найди свойство "{prop_name}" {(', означающее "' + self.task["desc"] + '"') if self.task["desc"] else ''} в приведённом контексте.
            """
            constraints = f""""""
            if self.possible_values:
                joined = "', '".join(self.possible_values)
                if len(joined) > 2048:
                    constraints += f"Учти, что значение искомого тобой свойства может быть только вида (список неполный): '{joined[:2048]}'\n"
                else:
                    constraints += f"Учти, что значение искомого тобой свойства может быть только одним из: '{joined}'\n"
            if self.possible_uoms:
                joined = "', '".join(self.possible_uoms)
                constraints += f"Учти, что единица измерения искомого тобой свойства может быть только одна из: '{joined}'\n"
        else:
            prompt = f"""{('Ты извлекаешь свойства для объекта "' + self.object_type + '".') if self.object_type else ''}
            Найди, упоминается ли свойство "{prop_name}", {('означающее "' + self.task["desc"] + '"') if self.task["desc"] else ''}
            в переданном {"тексте" if isinstance(src, str) else f"{gv.CURRENT_TABLE_REPR} представлении таблицы"} из корпоративного документа. Ты должен отвечать только да или нет.
            """

        if bypass_src:
            input_text = funcs.extracted_table_repr(src) if isinstance(src, ExtractedTable) else src
        else:
            input_text = funcs.get_frames(src, points)
        if len(input_text) < len("Отрывки: Таблицы: "):
            return
        prompt = re.sub(r"\s+", ' ', prompt)
        input_text = re.sub(r"\s+", ' ', input_text)
        system_messages = [prompt]
        if constraints:
            system_messages.append(re.sub(r"\s+", ' ', constraints))
        if self.desc_for_llm:
            system_messages.append(self.desc_for_llm)
        res_dict = gv.smart_llm_request(system_messages, input_text)

        val, uom = res_dict['value'], res_dict['uom']
        if self.condition == "only_if_mentioned":
            if val.startswith("д") or val.startswith("y"):
                self.result = "yes"
            return
        if isinstance(val, str):
            val = val.replace('\n', '')
        if isinstance(uom, str):
            uom = uom.replace('\n', '')
        self.result, self.uom = val, uom

    # TODO: rework without LLM
    def clarify_ans(self):
        pass

    @func_props(confidence=0.05, group=0)
    def search_in_tech_specs(self):
        if not self.tech_specs_frame:
            return
        self.execute_task(self.tech_specs_frame, 1, [0], bypass_src=True)

    @func_props(confidence=0.25, group=1)
    def tokens_search_ru(self):
        start = datetime.now()

        tokens_ru = gv.nlp_ru(self.task["ru_name"])
        best_ru = funcs.get_most_specific_tokens(self.pure_text, tokens_ru)
        if best_ru[1] != 10 ** 6:
            occurs_ru = funcs.find_all_occurs(self.pure_text, " ".join(
                [tokens_ru[i].text for i in range(len(tokens_ru))]))

            gv.METHODS_TIME_SPENT += (datetime.now() - start).total_seconds()

            if len(occurs_ru) < 50:
                self.execute_task(self.pure_text, 2, occurs_ru)
        else:
            gv.METHODS_TIME_SPENT += (datetime.now() - start).total_seconds()
    @func_props(confidence=0.30, group=1)
    def tokens_search_en(self):
        start = datetime.now()

        tokens_en = gv.nlp_en(self.task["en_name"])
        best_en = funcs.get_most_specific_tokens(self.pure_text, tokens_en)
        if best_en[1] != 10 ** 6:
            occurs_en = funcs.find_all_occurs(self.pure_text, " ".join(
                [tokens_en[i].text for i in range(len(tokens_en))]))

            gv.METHODS_TIME_SPENT += (datetime.now() - start).total_seconds()

            if len(occurs_en) < 50:
                self.execute_task(self.pure_text, 1, occurs_en)
        else:
            gv.METHODS_TIME_SPENT += (datetime.now() - start).total_seconds()


    @func_props(confidence=0.5, group=2)
    def simple_search_ru(self):
        start = datetime.now()

        simple_ru = funcs.find_all_occurs(self.pure_text, self.task["ru_name"], lemmatize=True)

        gv.METHODS_TIME_SPENT += (datetime.now() - start).total_seconds()

        if len(simple_ru) < 50:
            self.execute_task(self.pure_text, 2, simple_ru)
    @func_props(confidence=0.6, group=2)
    def simple_search_en(self):
        start = datetime.now()

        simple_en = funcs.find_all_occurs(self.pure_text, self.task["en_name"], lemmatize=True)

        gv.METHODS_TIME_SPENT += (datetime.now() - start).total_seconds()

        if len(simple_en) < 50:
            self.execute_task(self.pure_text, 1, simple_en)

    @func_props(confidence=0.8, group=3)
    def table_search_ru(self):
        start = datetime.now()

        points = funcs.search_in_tables(self.tables, self.task["ru_name"])

        gv.METHODS_TIME_SPENT += (datetime.now() - start).total_seconds()

        if points:
            if len(points) < 50:
                self.execute_task(self.tables, 1, points)
    @func_props(confidence=0.85, group=3)
    def table_search_en(self):
        start = datetime.now()

        points = funcs.search_in_tables(self.tables, self.task["en_name"])

        gv.METHODS_TIME_SPENT += (datetime.now() - start).total_seconds()

        if points:
            if len(points) < 50:
                self.execute_task(self.tables, 1, points)
