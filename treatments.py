"""Condition knowledge base + 30-day care schedule generator.
Advice is general guidance; always follow local agriculture-extension rules for chemicals."""

HEALTHY_TIPS = [
    "Water at the base in the morning; avoid wetting the leaves.",
    "Give 6-8 hours of light and keep good air circulation between plants.",
    "Check leaves (top and underside) once a week for early spots or pests.",
    "Remove fallen leaves and debris from the soil surface.",
    "Feed with balanced fertilizer as per the plant's season; avoid over-fertilizing.",
]

_GENERIC_NOTE = "Use any fungicide/pesticide only as per label and local agriculture guidelines."

RULES = [
    (("mosaic", "virus", "curl"), {
        "category": "Viral", "severity": "High",
        "summary": "A viral infection that spreads through insects (aphids, whiteflies), tools and hands. It cannot be cured, only controlled.",
        "treatment": [
            "Isolate the plant immediately from other plants.",
            "Remove and destroy (do not compost) heavily infected leaves/plants.",
            "Control vectors: spray neem oil or insecticidal soap on aphids/whiteflies.",
            "Disinfect tools and wash hands after handling the plant.",
            "Keep the plant well-watered and fed so new growth stays strong.",
        ],
        "prevention": ["Use virus-free seeds/seedlings.", "Use yellow sticky traps or nets.", "Control weeds that host insects."],
    }),
    (("mite",), {
        "category": "Pest", "severity": "Medium",
        "summary": "Tiny spider mites suck sap from leaves causing speckling, yellowing and webbing.",
        "treatment": [
            "Spray the plant with a strong jet of water, including leaf undersides.",
            "Apply neem oil or insecticidal soap every 5-7 days for 3 weeks.",
            "Remove badly damaged leaves.",
            "Raise humidity; mites thrive in hot, dry conditions.",
        ],
        "prevention": ["Inspect leaf undersides weekly.", "Avoid water stress and dust build-up."],
    }),
    (("greening", "haunglongbing"), {
        "category": "Bacterial (insect-spread)", "severity": "High",
        "summary": "Citrus greening is a serious bacterial disease spread by psyllids; there is no known cure.",
        "treatment": [
            "Report to your local agriculture office - it is a regulated disease in many regions.",
            "Remove and destroy infected trees to protect others.",
            "Control psyllids with recommended insecticides/oils.",
            "Support the tree with balanced nutrition while awaiting expert advice.",
        ],
        "prevention": ["Buy certified disease-free plants.", "Monitor new flushes for psyllids."],
    }),
    (("powdery",), {
        "category": "Fungal", "severity": "Medium",
        "summary": "A white powdery fungal coating on leaves that reduces photosynthesis.",
        "treatment": [
            "Remove heavily coated leaves and dispose of them in the trash.",
            "Spray a baking-soda solution (1 tsp/litre + few drops of soap) or sulfur/potassium bicarbonate fungicide.",
            "Improve airflow by pruning crowded growth.",
            "Repeat every 7 days until new growth stays clean.",
        ],
        "prevention": ["Avoid overhead watering.", "Give plants full sun and spacing."],
    }),
    (("rust",), {
        "category": "Fungal", "severity": "Medium",
        "summary": "Orange/brown pustules on leaves caused by rust fungi.",
        "treatment": [
            "Pick off and destroy affected leaves.",
            "Apply a copper- or sulfur-based fungicide every 7-10 days.",
            "Water only at the base and keep foliage dry.",
            "Clear fallen leaves around the plant.",
        ],
        "prevention": ["Choose resistant varieties.", "Avoid dense planting."],
    }),
    (("scab",), {
        "category": "Fungal", "severity": "Medium",
        "summary": "Dark, scabby lesions on leaves and fruit caused by a fungus.",
        "treatment": [
            "Remove infected leaves and fruit; rake fallen leaves.",
            "Apply a protectant fungicide (captan/copper) as per label during wet weather.",
            "Prune to open up the canopy for drying.",
        ],
        "prevention": ["Clean up leaf litter every autumn.", "Plant scab-resistant varieties."],
    }),
    (("rot", "esca"), {
        "category": "Fungal", "severity": "High",
        "summary": "Rot-causing fungi damage leaves, fruit and wood, often spreading in warm, humid weather.",
        "treatment": [
            "Prune out infected wood/leaves and mummified fruit; destroy them.",
            "Apply copper or a labelled fungicide on a 10-14 day cycle.",
            "Improve drainage and canopy airflow.",
            "Sterilize pruning tools between cuts.",
        ],
        "prevention": ["Remove old fruit and debris.", "Avoid wounding the plant."],
    }),
    (("scorch",), {
        "category": "Environmental / Fungal", "severity": "Medium",
        "summary": "Brown, dry leaf edges due to water stress, heat, salt build-up or infection.",
        "treatment": [
            "Water deeply and regularly; avoid drought stress.",
            "Mulch to keep roots cool and moist.",
            "Trim scorched leaves and flush soil if salts have built up.",
            "Provide afternoon shade during heat waves.",
        ],
        "prevention": ["Avoid over-fertilizing.", "Maintain consistent watering."],
    }),
    (("mold",), {
        "category": "Fungal", "severity": "Medium",
        "summary": "Leaf mold thrives in high humidity and shows yellow patches with mold underneath.",
        "treatment": [
            "Remove affected leaves; lower humidity and increase ventilation.",
            "Space plants and prune lower foliage.",
            "Apply a copper or labelled fungicide every 7-10 days.",
            "Water at soil level in the morning.",
        ],
        "prevention": ["Keep humidity below ~85%.", "Stake plants to improve airflow."],
    }),
    (("blight",), {
        "category": "Fungal / Oomycete", "severity": "High",
        "summary": "Blight causes fast-spreading brown lesions and leaf death, especially in cool, wet weather.",
        "treatment": [
            "Remove and destroy infected leaves immediately; do not compost them.",
            "Apply copper-based or labelled fungicide every 7 days during wet weather.",
            "Stop overhead watering and keep foliage dry.",
            "Stake/prune for airflow and mulch to prevent soil splash.",
        ],
        "prevention": ["Rotate crops yearly.", "Use resistant varieties and certified seed."],
    }),
    (("spot", "target", "septoria", "cercospora"), {
        "category": "Fungal / Bacterial", "severity": "Medium",
        "summary": "Circular spots on leaves that can merge and cause leaf drop.",
        "treatment": [
            "Remove spotted leaves and keep them away from the plant.",
            "Apply copper-based spray (works for both bacterial and fungal spots) every 7-10 days.",
            "Avoid working with the plant when leaves are wet.",
            "Mulch to prevent soil splash.",
        ],
        "prevention": ["Rotate crops.", "Water at the base; keep leaves dry."],
    }),
]

_DEFAULT = {
    "category": "Unspecified", "severity": "Medium",
    "summary": "A plant disease or stress was detected. Treat conservatively and confirm with an expert.",
    "treatment": [
        "Isolate the plant and remove visibly affected leaves.",
        "Water at the base and improve airflow and light.",
        "Consult a local agriculture expert with a clear photo for a confirmed diagnosis.",
    ],
    "prevention": ["Inspect plants weekly.", "Keep tools and pots clean."],
}


def clean(text: str) -> str:
    return text.replace("_", " ").replace("(", "").replace(")", "").strip().title()


def get_condition(class_name: str) -> dict:
    plant, _, cond = class_name.partition("___")
    base = {"plant": clean(plant), "condition": clean(cond or "Unknown")}
    low = cond.lower()
    if "healthy" in low:
        return {**base, "condition": "Healthy", "is_healthy": True, "category": "None",
                "severity": "None", "summary": "No disease detected.",
                "treatment": [], "prevention": HEALTHY_TIPS}
    for keys, info in RULES:
        if any(k in low for k in keys):
            data = {**info}
            break
    else:
        data = {**_DEFAULT}
    data["treatment"] = data["treatment"] + [_GENERIC_NOTE]
    return {**base, **data, "is_healthy": False}


def build_schedule(info: dict) -> list:
    """Return 30 day-entries: {day, phase, tasks[]}."""
    treat_days = {2, 6, 10, 16, 23}
    review_days = {7, 14, 21, 28}
    steps = info["treatment"]
    schedule = []
    for d in range(1, 31):
        if d <= 7:
            phase = "Phase 1 - Intensive treatment"
        elif d <= 14:
            phase = "Phase 2 - Control"
        elif d <= 21:
            phase = "Phase 3 - Recovery"
        else:
            phase = "Phase 4 - Prevention"

        if d == 1:
            tasks = ["Isolate the plant and remove all visibly infected leaves.",
                     "Take a clear baseline photo of the affected leaves.",
                     "Clean and disinfect your tools."]
        elif d in treat_days:
            tasks = [f"Treatment round: {steps[(sorted(treat_days).index(d)) % max(1, len(steps) - 1)]}",
                     "Check leaf undersides for new spots."]
        elif d in review_days:
            tasks = ["Weekly review: compare today's leaves with your day-1 photo.",
                     "Remove any newly infected leaves."]
        elif d == 30:
            tasks = ["Final assessment: upload a photo and record the final health score.",
                     "Decide whether to continue prevention care."]
        elif d <= 14:
            tasks = ["Inspect all leaves and remove new infected parts.",
                     "Water at the base only; keep leaves dry."]
        elif d <= 21:
            tasks = ["Check for new growth and note any new symptoms.",
                     "Maintain light, airflow and balanced watering."]
        else:
            tasks = ["Quick leaf inspection.",
                     "Keep soil moist but not soggy; no overhead watering."]
        schedule.append({"day": d, "phase": phase, "tasks": tasks})
    return schedule