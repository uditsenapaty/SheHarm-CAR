#!/usr/bin/env python3
"""Hand-authored seed inventory for the Women-Harm Knowledge Ontology.

Kept separate from the builder so the knowledge content stays reviewable as flat lists,
while `build_ontology.py` owns composition, triple generation, and count validation.

Target counts (paper Table `tab:ontology-statistics`):
    women-related target concepts   126
    harm-cue concepts               438
    harm-category concepts            5
    context and exception concepts   42
    ------------------------------------
    total concepts                  611
    relation types                   14
    ontology triples              1,287
    soft reasoning rules             36
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# 1. Women-related target concepts (126), grouped by ontology target type
# --------------------------------------------------------------------------

TARGET_CONCEPTS: dict[str, list[str]] = {
    "Female-Role": [
        "woman", "women", "girl", "girls", "lady", "ladies", "female", "females",
        "young woman", "older woman", "teenage girl", "adult woman", "woman passenger",
        "woman in public", "woman on the street", "female stranger", "female speaker",
        "female protagonist", "female character", "female cartoon character",
        "female celebrity", "female influencer", "female model",
    ],
    "Female-Relationship": [
        "wife", "ex-wife", "girlfriend", "ex-girlfriend", "fiancee", "bride",
        "mother", "stepmother", "mother-in-law", "grandmother", "daughter",
        "stepdaughter", "sister", "sister-in-law", "aunt", "niece", "female cousin",
        "female partner", "female friend", "female neighbour", "single mother",
        "pregnant woman", "housewife", "homemaker", "divorced woman", "widow",
    ],
    "Female-Profession": [
        "female politician", "female leader", "female executive", "female manager",
        "female engineer", "female scientist", "female doctor", "nurse",
        "female teacher", "female professor", "female student", "female lawyer",
        "female judge", "female journalist", "female news anchor", "actress",
        "female singer", "female athlete", "female driver", "female pilot",
        "female soldier", "female police officer", "female cashier", "waitress",
        "female flight attendant", "female receptionist", "female secretary",
        "female programmer", "female gamer", "female streamer",
    ],
    "Female-Appearance": [
        "her face", "her body", "her hair", "her skin", "her weight", "her figure",
        "her makeup", "her clothing", "her dress", "her outfit", "her legs",
        "her chest", "her age", "her height", "her nose", "her teeth",
        "her body shape", "her natural look", "her cosmetic surgery",
        "her physical appearance",
    ],
    "Female-Group": [
        "women as a group", "all women", "modern women", "feminists",
        "women drivers", "women shoppers", "female employees", "female colleagues",
        "women in politics", "women in sports", "women in tech", "girlfriends",
        "wives", "mothers", "housewives as a group", "female fans",
    ],
    "Female-Behaviour": [
        "woman driving", "woman shopping", "woman talking", "woman cooking",
        "woman cleaning", "woman working", "woman parking", "woman arguing",
        "woman crying", "woman dressing up", "woman taking selfies",
    ],
}

# --------------------------------------------------------------------------
# 2. Harm-cue concepts (438), grouped by the harm category they express
# --------------------------------------------------------------------------

HARM_CUES: dict[str, list[str]] = {
    "Sexual-Harassment": [
        "sexual comment", "sexualized joke", "sexual innuendo",
        "lewd comment", "explicit sexual proposition", "unwanted sexual advance",
        "catcalling", "wolf whistle", "street harassment", "sexual objectification",
        "reduces woman to body parts", "body as sexual object", "leering",
        "ogling", "staring at her chest", "unsolicited sexual image",
        "sexual demand", "sexual coercion", "sexual blackmail", "rape threat",
        "rape joke", "sexual assault reference", "groping reference",
        "non-consensual touching", "forced intimacy", "sexual entitlement",
        "she owes him sex", "sex as payment", "transactional sex framing",
        "sexual availability assumption", "sexual conquest boast",
        "scoring with women", "sexual scorekeeping", "women as sexual reward",
        "trophy framing", "sexual rating of women", "rating her body",
        "hotness scale", "sexual comparison of women", "pornographic framing",
        "sexualizing a professional", "sexualizing a minor-coded figure",
        "sexualizing an athlete", "sexualizing a colleague",
        "sexual harassment at work", "boss sexual pressure",
        "sexual favour for promotion", "casting couch reference",
        "sexual objectification in advertising", "upskirt reference",
        "voyeurism", "hidden camera reference", "revenge porn reference",
        "leaked intimate image", "sextortion", "sexual humiliation",
        "public sexual shaming", "forced exposure", "clothing removal demand",
        "sexual gesture", "obscene gesture", "genital reference",
        "breast reference as insult", "sexual slur", "sexual name-calling",
        "sexual stereotype of women", "women as sexually available",
        "sexual double standard", "sexual purity policing",
        "virginity policing", "sexual experience shaming",
        "sexual harassment normalization", "harassment as compliment",
        "she should take it as a compliment", "consent dismissal",
        "no means yes framing", "persistence after refusal",
        "stalking reference", "unwanted pursuit", "sexual intimidation",
        "sexual threat of violence", "sexual degradation",
        "sexual servitude framing", "sexual ownership claim",
        "she belongs to him sexually", "marital sexual entitlement",
        "sexual harassment minimization", "boys will be boys defence",
        "sexual predation humour", "predatory framing", "sexual grooming reference",
        "sexual exploitation reference", "trafficking reference",
        "sexual harassment victim blaming", "she asked for it",
        "her clothing invited it", "sexual harassment denial",
    ],
    "Violence": [
        "beat her", "hit her", "slap her", "punch her", "kick her", "strangle her",
        "choke her", "kill her", "murder threat", "death threat", "stab reference",
        "shoot her", "burn her", "acid attack", "acid throwing reference",
        "honour killing reference", "dowry violence", "bride burning",
        "domestic violence", "domestic abuse joke", "wife beating",
        "wife beating humour", "partner violence", "intimate partner abuse",
        "physical punishment of wife", "discipline through violence",
        "she needs a beating", "violence as correction", "violence as discipline",
        "threat of physical harm", "intimidation with violence",
        "encouragement of violence", "incitement to attack",
        "calls for physical harm", "glorification of violence",
        "celebrating violence against women", "violent imagery",
        "visual injury", "bruises depicted", "blood depicted", "weapon aimed at her",
        "gun pointed at woman", "knife threat", "belt as weapon",
        "kitchen violence reference", "cooking pan violence trope",
        "violent restraint", "tied up depiction", "forced confinement",
        "kidnapping reference", "abduction humour", "hostage framing",
        "violence normalization", "violence minimization",
        "she provoked the violence", "violence victim blaming",
        "self-defence inversion", "mutual abuse framing", "revenge violence",
        "retaliation threat", "punishment threat", "harm to pregnant woman",
        "violence during pregnancy", "harm to mother", "harm to daughter",
        "threat to family members", "threat to children",
        "violence in relationship breakup", "post-breakup threat",
        "stalking with threat", "violent stalking", "assault reference",
        "battery reference", "grievous injury reference", "torture reference",
        "mutilation reference", "disfigurement threat", "throwing objects at her",
        "pushing her", "shoving reference", "dragging reference",
        "hair pulling", "physical humiliation", "public beating",
        "mob violence against women",
    ],
    "Misogyny": [
        "belongs in the kitchen", "kitchen stereotype", "make me a sandwich",
        "women should cook", "women should clean", "women belong at home",
        "domestic duty stereotype", "housework is her job",
        "women cannot drive", "bad woman driver stereotype", "parking joke",
        "women are bad at maths", "women cannot do science",
        "women are not logical", "women are emotional", "too emotional to lead",
        "hormonal dismissal", "must be her period", "irrational woman trope",
        "women talk too much", "nagging wife trope", "gossiping women trope",
        "women cannot keep secrets", "women are gold diggers",
        "after his money", "expensive wife trope", "shopping addiction trope",
        "women waste money", "women are dependent", "cannot survive alone",
        "needs a man", "women are inferior", "second-class framing",
        "male superiority claim", "men are naturally better",
        "women are property", "ownership of wife", "obedience expectation",
        "wife must obey", "submission expectation", "know your place",
        "women should be silent", "silencing women", "opinion dismissal",
        "women's opinion is worthless", "mansplaining framing",
        "female incompetence claim", "hire based on looks",
        "diversity hire slur", "she slept her way up", "token woman claim",
        "women in tech mockery", "women in sports mockery",
        "women in politics mockery", "female leader mockery",
        "women cannot lead", "unfit for leadership", "exclusion from work",
        "workplace exclusion", "pay gap justification", "women deserve less pay",
        "maternity as burden", "pregnancy as liability",
        "motherhood mandate", "childless woman shaming",
        "biological clock pressure", "marriage pressure",
        "women expire with age", "expiry date trope", "value tied to youth",
        "value tied to fertility", "value tied to virginity",
        "feminism mockery", "feminist ridicule", "equality rejection",
        "women's rights dismissal", "gender role enforcement",
        "traditional role enforcement", "women are manipulative",
        "women are deceitful", "women lie about assault",
        "false accusation trope", "women weaponize tears",
        "female privilege claim", "reverse sexism claim",
        "women cannot be funny", "female driver insurance joke",
        "back to the kitchen command", "sandwich command",
        "women as servants", "servitude expectation", "wife as maid",
        "wife as cook", "unpaid labour expectation",
        "gender stereotype reinforcement", "essentialist gender claim",
        "women are all the same", "generalization about women",
        "dehumanizing generalization", "women as objects",
        "women as decoration", "trophy wife framing",
    ],
    "Appearance-Attack": [
        "too ugly", "she is ugly", "ugly insult", "hideous remark",
        "unattractive insult", "face mockery", "face insult",
        "mocking her smile", "mocking her teeth", "mocking her nose",
        "mocking her eyes", "mocking her ears", "mocking her chin",
        "mocking her forehead", "skin mockery", "acne mockery",
        "wrinkle mockery", "ageing mockery", "old and ugly framing",
        "grey hair mockery", "hair mockery", "bald spot mockery",
        "hairstyle ridicule", "body shaming", "fat shaming", "calling her fat",
        "whale comparison", "pig comparison", "cow comparison",
        "animal comparison insult", "weight mockery", "obesity insult",
        "skinny shaming", "too thin insult", "flat chest mockery",
        "breast size mockery", "hip size mockery", "thigh mockery",
        "leg mockery", "arm mockery", "belly mockery", "stretch mark mockery",
        "cellulite mockery", "posture mockery", "height mockery",
        "too short insult", "too tall insult", "makeup mockery",
        "without makeup mockery", "catfish accusation",
        "before and after mockery", "filter accusation",
        "photoshop accusation", "plastic surgery mockery",
        "botox mockery", "lip filler mockery", "cosmetic procedure ridicule",
        "fake beauty accusation", "clothing mockery", "outfit ridicule",
        "dress shaming", "revealing clothing shaming",
        "conservative clothing mockery", "cheap clothing insult",
        "fashion sense insult", "shoe mockery", "accessory mockery",
        "hygiene insult", "smell insult", "unkempt insult",
        "grooming insult", "eyebrow mockery", "nail mockery",
        "voice mockery", "laugh mockery", "beauty standard enforcement",
        "must be pretty to matter", "appearance rating", "rating her looks",
        "out of your league framing", "downgrade comparison",
        "appearance comparison between women", "beauty competition framing",
    ],
    "Character-Assassination": [
        "she is characterless", "characterless woman", "no morals claim",
        "immoral woman framing", "slut shaming", "slut label", "whore label",
        "prostitute accusation", "loose woman claim", "easy woman claim",
        "promiscuity accusation", "body count shaming", "sleeping around claim",
        "cheating accusation", "unfaithful wife claim", "adultery accusation",
        "home wrecker label", "mistress label", "other woman blame",
        "gold digger accusation", "using men for money",
        "marriage for money claim", "divorce for money claim",
        "alimony scheme claim", "child support scheme claim",
        "false accusation claim", "lying about assault claim",
        "attention seeking claim", "drama queen label", "manipulative label",
        "toxic woman label", "crazy ex label", "psycho label",
        "unstable woman claim", "bad mother accusation",
        "neglectful mother claim", "unfit parent claim",
        "abandoned her children claim", "bad wife accusation",
        "lazy wife claim", "incompetent homemaker claim",
        "reputation attack", "public shaming", "character defamation",
        "moral judgement", "moral policing", "purity policing",
        "religious shaming", "cultural shaming", "family dishonour claim",
        "brings shame to family", "gossip about her",
        "rumour spreading", "social exclusion campaign",
        "trustworthiness attack", "dishonest woman claim",
        "fake personality claim", "two-faced label", "backstabber label",
        "opportunist label", "career through favours claim",
        "undeserved success claim", "credibility attack",
        "professional reputation attack", "questioning her qualifications",
        "questioning her integrity", "corruption insinuation",
        "greed insinuation", "selfishness accusation",
        "ungrateful woman claim", "entitled woman label",
    ],
}

# --------------------------------------------------------------------------
# 3. Harm-category concepts (5)
# --------------------------------------------------------------------------

CATEGORY_CONCEPTS: list[str] = [
    "Sexual-Harassment", "Violence", "Misogyny",
    "Appearance-Attack", "Character-Assassination",
]

# --------------------------------------------------------------------------
# 4. Context and exception concepts (42)
# --------------------------------------------------------------------------

CONTEXT_CONCEPTS: dict[str, list[str]] = {
    "Counter-Speech": [
        "counter-speech", "criticism-of-misogyny", "condemnation",
        "calling out sexism", "challenging a stereotype", "rebuttal to harm",
        "solidarity with women", "support for survivors", "allyship statement",
    ],
    "Quotation": [
        "quotation", "quoted harmful phrase", "screenshot of abuse",
        "reporting someone else's words", "depicting harm to expose it",
        "documenting harassment", "attributed statement",
    ],
    "Awareness": [
        "awareness", "awareness campaign", "educational framing",
        "public-service message", "statistics about abuse",
        "helpline information", "prevention message", "survivor testimony",
    ],
    "Negation": [
        "negation cue", "explicit rejection", "denial of stereotype",
        "correction of a claim", "refutation", "disagreement marker",
    ],
    "Empowerment": [
        "empowerment message", "celebration of achievement",
        "praise for a woman", "role-model framing", "equality affirmation",
        "respect statement",
    ],
    "Non-Targeted": [
        "self-directed humour", "general humour without target",
        "male-directed content", "non-gendered subject",
        "animal or object subject", "neutral observation",
    ],
}

# --------------------------------------------------------------------------
# 5. Relation types (14)
# --------------------------------------------------------------------------

RELATIONS: list[dict[str, str]] = [
    {"name": "is_women_related_role", "domain": "target", "range": "target_type"},
    {"name": "is_a", "domain": "concept", "range": "concept_type"},
    {"name": "expresses", "domain": "harm_cue", "range": "harm_category"},
    {"name": "indicates", "domain": "harm_cue", "range": "harmfulness"},
    {"name": "supports", "domain": "context", "range": "harmfulness"},
    {"name": "negated_by", "domain": "harm_category", "range": "context"},
    {"name": "quoted_in", "domain": "harm_category", "range": "context"},
    {"name": "supported_by", "domain": "harm_category", "range": "evidence"},
    {"name": "mitigates", "domain": "context", "range": "harm_category"},
    {"name": "escalates", "domain": "harm_cue", "range": "harm_category"},
    {"name": "co_occurs_with", "domain": "harm_category", "range": "harm_category"},
    {"name": "targets", "domain": "harm_category", "range": "target_type"},
    {"name": "refers_to", "domain": "target", "range": "modality"},
    {"name": "evidenced_by", "domain": "target_type", "range": "evidence"},
]

EVIDENCE_CONCEPTS = ["Textual Evidence", "Visual Evidence"]
MODALITIES = ["visual", "textual", "multimodal"]
HARMFULNESS = ["Explicit-Harm", "Implicit-Harm", "Non-Harm"]

# Cues whose realization is typically indirect (stereotype, implication, sarcasm).
# Everything else defaults to Explicit-Harm for the `indicates` relation.
IMPLICIT_CATEGORIES = {"Misogyny"}
IMPLICIT_MARKERS = (
    "stereotype", "trope", "joke", "humour", "framing", "claim", "assumption",
    "insinuation", "mockery", "ridicule", "comparison", "normalization",
    "minimization", "dismissal", "expectation", "pressure", "generalization",
    "label", "policing", "enforcement", "reference",
)
