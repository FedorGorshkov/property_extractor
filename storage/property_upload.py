import json

import src.global_variables as gv

hardcode = json.loads(open("hardcode_config.json").read())
SEARCH_IN_HEADER = hardcode["SEARCH_IN_HEADER"]
ONLY_IF_MENTIONED = hardcode["ONLY_IF_MENTIONED"]
ALT_KEYWORDS = hardcode["ALT_KEYWORDS"]
DESC_FOR_LLM = hardcode["DESC_FOR_LLM"]
TRY_BYPASSING_JUST_OBJ_TYPE = hardcode["TRY_BYPASSING_JUST_OBJ_TYPE"]
props = json.loads(open("dumpObjects.json").read()) + json.loads(open("dumpDatatypes.json").read())
new_props, seen = [], []
for prop in props:
    if (( "RU" in prop["names"] ) and ( "EN" in prop["names"] ) and
            ( ".ru" not in prop["id"] ) and ( ".org" not in prop["id"] ) and ('-' in prop["id"])):
        if prop["names"]["RU"] and prop["names"]["EN"]:
            if prop["names"]["EN"] in seen:
                continue
            desc_en, desc_ru, condition, alt_keywords = 'none', 'none', 'none', 'none'
            if "EN" in prop["comments"] and "RU" in prop["comments"]:
                desc_en, desc_ru = prop["comments"]["EN"], prop["comments"]["RU"]
            if prop["names"]["EN"] in SEARCH_IN_HEADER:
                condition = "search_in_header"
            elif prop["names"]["EN"] in ONLY_IF_MENTIONED:
                condition = "only_if_mentioned"
            if prop["names"]["EN"] in ALT_KEYWORDS:
                alt_keywords = ALT_KEYWORDS[prop["names"]["EN"]]
            flag = 0
            if prop["names"]["EN"] in TRY_BYPASSING_JUST_OBJ_TYPE:
                flag = 1
            desc_for_llm = ""
            if prop["names"]["EN"] in DESC_FOR_LLM:
                desc_for_llm = DESC_FOR_LLM[prop["names"]["EN"]]
            new_props.append((prop["id"], prop["names"]["EN"], prop["names"]["RU"], desc_en, desc_ru, condition, alt_keywords, desc_for_llm, flag))
            seen.append(prop["names"]["EN"])

added, updated = 0, 0
for i in range(len(new_props)):
    if gv.cur.execute("SELECT * FROM properties WHERE prop_name_eng = ?", (new_props[i][1], )).fetchone():
        gv.cur.execute("UPDATE properties SET external_id = ?, prop_name_rus = ?, prop_desc_eng = ?, prop_desc_rus = ?, condition = ?, alt_keywords = ?, desc_for_llm = ?, flag = ? WHERE prop_name_eng = ?",
                       (new_props[i][0], new_props[i][2], new_props[i][3], new_props[i][4], new_props[i][5], new_props[i][6], new_props[i][7], new_props[i][8], new_props[i][1]))
        updated += 1
        continue
    gv.cur.execute("INSERT INTO properties (external_id, prop_name_eng, prop_name_rus, prop_desc_eng, prop_desc_rus, condition, alt_keywords, desc_for_llm, flag) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", new_props[i])
    added += 1

print(f"Added {added} new properties, updated {updated} existing ones")
