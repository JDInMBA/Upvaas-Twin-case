"""Indian food library used by the meal simulator and the What-if sandbox.

carbs  : available carbohydrate per typical serving (g)
gi     : glycaemic index (glucose = 100)
fat    : 0-1 "fat/fibre slowing" factor; fried or fatty food empties the stomach more slowly

Sources: ragi mudde 98, jowar roti 84, bajra roti 70, wheat dosa 62, upma 52, chickpea sundal 24
are measured values (IJMR, "Carbohydrate profiling & glycaemic indices of selected traditional
Indian foods"). All others are approximate [assumption] and should be replaced with values from
the Indian Food Composition Tables / published GI studies before clinical use.
"""
from __future__ import annotations

FOODS: dict[str, dict] = {
    # --- everyday --------------------------------------------------------------
    "masala_chai_sugar": {"label": "Masala chai (2 tsp sugar)", "carbs": 12, "gi": 65, "fat": 0.1},
    "marie_biscuits":    {"label": "Marie biscuits x3", "carbs": 15, "gi": 70, "fat": 0.2},
    "poha":              {"label": "Poha (1 plate)", "carbs": 45, "gi": 70, "fat": 0.2},
    "idli_sambar":       {"label": "Idli x3 + sambar", "carbs": 50, "gi": 75, "fat": 0.1},
    "wheat_dosa":        {"label": "Wheat dosa + chutney", "carbs": 40, "gi": 62, "fat": 0.3},
    "upma":              {"label": "Upma", "carbs": 40, "gi": 52, "fat": 0.3},
    "aloo_paratha":      {"label": "Aloo paratha x2 + curd", "carbs": 70, "gi": 65, "fat": 0.5},
    "rice_dal_sabzi":    {"label": "Rice (1.5 cup) + dal + sabzi", "carbs": 85, "gi": 72, "fat": 0.2},
    "roti_dal_sabzi":    {"label": "Roti x3 + dal + sabzi", "carbs": 65, "gi": 62, "fat": 0.25},
    "ragi_mudde":        {"label": "Ragi mudde + saaru", "carbs": 60, "gi": 98, "fat": 0.1},
    "jowar_roti_sabzi":  {"label": "Jowar roti x2 + sabzi", "carbs": 50, "gi": 84, "fat": 0.2},
    "bajra_roti_sabzi":  {"label": "Bajra roti x2 + sabzi", "carbs": 50, "gi": 70, "fat": 0.25},
    "veg_biryani":       {"label": "Biryani (1 plate)", "carbs": 90, "gi": 70, "fat": 0.45},
    "chickpea_sundal":   {"label": "Chickpea sundal", "carbs": 25, "gi": 24, "fat": 0.2},
    "samosa":            {"label": "Samosa x1", "carbs": 25, "gi": 60, "fat": 0.6},
    "banana":            {"label": "Banana", "carbs": 27, "gi": 55, "fat": 0.0},
    # --- fasting (vrat / roza) ---------------------------------------------------
    "sabudana_khichdi":  {"label": "Sabudana khichdi", "carbs": 75, "gi": 85, "fat": 0.35},
    "kuttu_puri_aloo":   {"label": "Kuttu puri x3 + aloo", "carbs": 60, "gi": 55, "fat": 0.6},
    "singhara_halwa":    {"label": "Singhara atta halwa", "carbs": 50, "gi": 70, "fat": 0.5},
    "makhana_roasted":   {"label": "Roasted makhana", "carbs": 15, "gi": 40, "fat": 0.2},
    "fruit_plate":       {"label": "Fruit plate", "carbs": 30, "gi": 50, "fat": 0.0},
    "sweet_gulab_jamun": {"label": "Gulab jamun x2", "carbs": 45, "gi": 75, "fat": 0.5},
    "sweet_jalebi":      {"label": "Jalebi (100 g)", "carbs": 50, "gi": 85, "fat": 0.4},
    "kheer":             {"label": "Kheer (1 bowl)", "carbs": 40, "gi": 55, "fat": 0.4},
    "sargi_platter":     {"label": "Sargi: mathri, feni, fruit, dry fruit", "carbs": 65, "gi": 70, "fat": 0.5},
    "dates_x3":          {"label": "Dates x3", "carbs": 54, "gi": 45, "fat": 0.0},
    "rooh_afza_sharbat": {"label": "Rose sharbat (1 glass)", "carbs": 30, "gi": 90, "fat": 0.0},
    "pakora_plate":      {"label": "Pakora plate", "carbs": 35, "gi": 60, "fat": 0.7},
    "sehri_paratha_egg": {"label": "Sehri: paratha + egg + milk", "carbs": 55, "gi": 60, "fat": 0.5},
    "festive_thali":     {"label": "Festive thali + sweets", "carbs": 110, "gi": 72, "fat": 0.5},
}


def absorption_rate(gi: float, fat: float) -> float:
    """Map GI and fat to an intestinal absorption constant k_abs (1/min).

    High GI -> fast absorption (sharp spike); fat slows gastric emptying. Range ~0.006-0.028/min,
    centred on the Dalla Man T2D value k_abs = 0.023/min [assumption: the GI mapping itself].
    """
    k = 0.006 + 0.00022 * gi
    return k * (1.0 - 0.45 * fat)


BREAKFASTS = ["poha", "idli_sambar", "wheat_dosa", "upma", "aloo_paratha"]
LUNCHES = ["rice_dal_sabzi", "roti_dal_sabzi", "ragi_mudde", "jowar_roti_sabzi", "veg_biryani"]
DINNERS = ["roti_dal_sabzi", "rice_dal_sabzi", "bajra_roti_sabzi", "jowar_roti_sabzi"]
SNACKS = ["marie_biscuits", "samosa", "banana", "chickpea_sundal"]
