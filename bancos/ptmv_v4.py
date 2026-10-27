"""
Banco PTM-V v4 (oct 2026) y prompt de puntuacion v4.

Cambios respecto de v3, a partir del corpus de 20 (22 casos, modo libro):

  N/A inconsistente (misma unidad: N/A en una corrida y numero en otra), meta <= 0.15
    P5 0.34 · P2 0.32 · Vp3 0.29 · Vp2 0.29 · P4 0.23 · Vr2 0.20 · Vp4 0.20 · P1 0.19 · P3 0.19
  (el notebook mostraba el doble: agrupaba fragmentos de obras distintas como si
  fueran la misma unidad; corregido en litfem 0.15.0)

  Causa (leyendo los razonamientos discordantes): la situacion SI esta en el
  fragmento, pero el modelo duda si alcanza el umbral del enunciado
  ("not overt", "not coercive of the kind required", "no explicit label like
  'weak'"). Los ejemplos fuertes ('monster', 'servant') y palabras como
  "oppression" suben ese umbral.

  Arreglo:
    1. P1-P5, Vr2: el disparador incluye explicitamente las formas leves,
       indirectas o implicitas; el N/A queda para la ausencia TOTAL.
    2. Vp2: se quita "public or collective" del enunciado (chocaba con su
       propia condicion de N/A, "speaks to nobody").
    3. Vp3, Vp4: las decisiones familiares y las convenciones sociales
       (matrimonio, dinero, reputacion) cuentan como espacio/norma. Es un
       cambio de constructo menor: revisarlo antes de correr.
    4. Vp1: la posicion social se infiere de cualquier indicio.
    5. Prompt: regla de umbral en el paso 1 ("en la duda, Material: yes").

Las demas preguntas (T, M, Vn, Vr1, Vr3-Vr5, Vp5) quedan igual que en v3.
Cambia la firma de configuracion: usar una carpeta de salida nueva.
"""

PREGUNTAS_PTMV = [

    # ── PERSISTENCIA (P) ── mantenerse a sí mismo bajo presión
    ("P1", "In situations where maintaining their convictions, dignity or moral principles carries any cost, "
           "friction or risk — even a minor or indirect one, such as disapproval, inconvenience, a tempting "
           "shortcut or someone else's displeasure — the character prioritizes their integrity over convenience. "
           "If they give in for convenience, score low. Holding firm is a high score, not N/A. N/A only if the "
           "character makes no choice or takes no stance of any kind in the excerpt."),
    ("P2", "When others treat the character according to an expectation, role or judgment about who they are or "
           "should be — explicitly or implicitly, through words, rules, assumptions or simply the way they are "
           "treated (e.g. 'a child to be managed', 'a ruined woman', 'just a servant') — the character affirms or "
           "defends their own sense of self. If they accept or internalize the imposed view, score low. Rejecting "
           "it is a high score, not N/A. N/A only if no other character, group or social norm treats or judges "
           "the character in any way in this excerpt (the narrator's own description does not count)."),
    ("P3", "When the character faces any restriction, control, threat or adversity — including social disapproval, "
           "dependence on others, illness, poverty or hostile circumstances, not only overt oppression — they keep "
           "a will of their own and a coherent 'I', rather than being reduced to a passive victim of circumstances. "
           "Coping or resisting is a high score, not N/A. N/A only if the character faces no difficulty or "
           "constraint of any kind in this excerpt."),
    ("P4", "After obstacles, setbacks or changes — large or small, including disappointments, refusals or "
           "frustrated plans — the character maintains the direction of their goals or priorities instead of "
           "abandoning them without justification. Overcoming the obstacle is a high score, not N/A. N/A only if "
           "nothing goes wrong or changes for the character in this excerpt."),
    ("P5", "When others or circumstances push the character to do, accept or believe something — commands, rules, "
           "persuasion, expectations, emotional or social pressure all count, even mild ones — the character keeps "
           "their own judgment about it: either they resist openly, or they comply outwardly while dissenting "
           "inwardly. Open resistance and acting freely against the pressure count as full dissent (score high). "
           "If they internalize the imposition and give up their own judgment, score low. N/A only if nobody and "
           "nothing pushes the character in any direction in this excerpt."),

    # ── IMPACTO TRANSFORMADOR (T) ── mover la trama  (sin cambios)
    ("T1", "Within this excerpt, the character's actions act as the main engine driving events forward, acting on "
           "their own initiative rather than merely reacting passively or defensively to their environment. N/A only "
           "if the excerpt is purely descriptive and the character does not act at all."),
    ("T2", "Within this excerpt, the changes that occur are a direct result of the character's plans, decisions or "
           "explicit will, rather than luck, magic or rescue by others. N/A only if nothing changes in the excerpt."),
    ("T3", "By the end of this excerpt, the character's intervention has left the situation different from how it "
           "began, in a way that cannot simply be undone. Judge the change relative to the start of THIS excerpt, not "
           "to the novel as a whole. If actions occur but everything stays as it was, score low. N/A only if the "
           "character does not act or intervene at all in this excerpt."),
    ("T4", "The character actively succeeds in transforming the state of affairs beyond themselves, whether by "
           "changing other characters' beliefs or decisions, or by altering the physical, social or institutional "
           "environment. If they only reflect internally without affecting anyone or anything beyond themselves, "
           "score low. N/A only if the character has no opportunity to affect anyone or anything beyond themselves "
           "in this excerpt."),
    ("T5", "The character's participation is decisive for the outcome of the events narrated in THIS excerpt: without "
           "their intervention, the result would have been radically different or remained unresolved. N/A only if "
           "the character does not participate in the events narrated in this excerpt."),

    # ── INTELIGIBILIDAD MOTIVACIONAL (M) ── ser comprensible por dentro  (sin cambios)
    ("M1", "The snippet provides enough evidence (either explicit or suggested between lines) to understand the "
           "reasons behind the character's behavior, making intelligible the connection between their personal values "
           "and their actions. (If the character does not act or make decisions in the text, it is marked N/A)"),
    ("M2", "The reactions and decisions of the character within this fragment are consistent with the mentality and "
           "emotional state presented; they do not turn out to be arbitrary or change abruptly without a psychological "
           "justification that the text allows to understand."),
    ("M3", "The text reveals or suggests a complex inner life, hinting at deep motivations, doubts, or internal "
           "conflicts that go beyond a mere superficial impulse or a single flat emotion."),
    ("M4", "The narrative devotes attention to the character's goals, concerns or aspirations as motivations of their "
           "own, rather than treating them as someone who exists only to serve, love or oppose another character. "
           "Immediate concerns and practical wants count as goals of their own; long-term ambitions are not required."),
    ("M5", "Through their thoughts, dialogues or actions in this text, the character exhibits nuances and "
           "particularities that move them away from being a mere stereotype or a predictable cliché."),

    # ── VALIDACIÓN NARRATIVA (Vn) ── cómo lo trata la voz narrativa  (sin cambios)
    ("Vn1", "In this excerpt, the narrative voice treats the character's experiences, decisions, or state with respect, "
            "granting them gravity and dignity regardless of whether the scene describes a success, a failure, or a "
            "mundane moment."),
    ("Vn2", "In the evaluated excerpt, the text presents the character's internal world (thoughts, emotions, doubts) as "
            "a complex and valid experience, giving human meaning to their perspective. (Mark N/A if the excerpt is "
            "purely physical action/dialogue with no access to interiority)."),
    ("Vn3", "In the situations described in this excerpt, the narrative makes visible the influence of the environment, "
            "circumstances, or social barriers on the character's life, avoiding reducing their successes or failures "
            "to a simple isolated moral flaw or merit. (Mark N/A if there is no conflict or interaction with the "
            "environment in this passage)."),
    ("Vn4", "During this excerpt, the text endows the character with intrinsic worth and complexity, keeping it firm and "
            "independent of the prejudices, labels, or indifference imposed by the rest of the fictional society present."),
    ("Vn5", "The general tone of this fragment conveys the narrator's stance that the character deserves justice, "
            "well-being or compassion for what is happening to them at this moment. A humorous or light tone can still "
            "convey this: judge whether the narrator sides with the character, not the seriousness of the tone."),

    # ── VALIDACIÓN RELACIONAL (Vr) ── cómo lo tratan los otros personajes
    ("Vr1", "The character has at least one bond in which they are valued as a person, regardless of usefulness, role "
            "or status. If other characters are present but no such bond is shown, score low. N/A only if the "
            "character interacts with nobody in this excerpt."),
    ("Vr2", "When the character is in crisis, fear, pain, illness or vulnerability — even a moderate one, such as "
            "distress, confusion, loneliness or humiliation — they receive genuine care or shelter from others. If "
            "they are vulnerable and nobody cares for them, including because nobody is around, score 1. N/A only "
            "if the character shows no vulnerability of any kind in this excerpt."),
    ("Vr3", "Allies or people close to the character defend, support or actively help them through difficulties, rather "
            "than abandoning them. If the character has no allies here, or they fail to help, score low. N/A only if "
            "no other characters appear in the excerpt."),
    ("Vr4", "Others recognize the character's suffering or feelings as legitimate and important, without minimizing or "
            "invalidating them. If others dismiss or ignore them, score low. N/A only if the character expresses no "
            "feelings or interacts with nobody."),
    ("Vr5", "The character's close relationships show explicit affection and, when there is conflict or rupture, a real "
            "disposition towards reparation and understanding. If the relationships shown are cold, mocking or hostile, "
            "score low. N/A only if no other characters appear in the excerpt."),

    # ── VALIDACIÓN INSTITUCIONAL (Vp) ── cómo lo trata el mundo social
    ("Vp1", "The character holds a role, position or level of resources that grants them authority, status or privileges "
            "recognized by their environment. Infer their position from any cue (household, servants, dress, work, "
            "how others address them). If the character clearly holds no such status, score low. N/A only if the "
            "excerpt gives no cue at all about the character's social position."),
    ("Vp2", "The character's opinions, interventions or requests are taken seriously by the people around them, in "
            "any setting, private or public (family, household, social gatherings, institutions). If they are "
            "ignored, mocked, dismissed or overruled without consideration, score low. N/A only if the character "
            "expresses no opinion, request or intervention to anyone in this excerpt."),
    ("Vp3", "The character can take part in the decisions and opportunities that affect their life — family and "
            "household decisions (marriage, money, residence), work, education, social circles or institutions — "
            "without being excluded, kept uninformed or blocked by their status, gender, class or reputation. If "
            "decisions are made over their head or they are barred, score low. N/A only if no decision or "
            "opportunity affecting the character arises in this excerpt."),
    ("Vp4", "The norms, laws, authorities or conventions of the narrative world — including family authority, "
            "marriage, religion, reputation and social custom — operate in the character's favor, protecting them "
            "or backing their interests. If they operate against the character or ignore them, score low. N/A only "
            "if no norm, authority or convention bears on the character's situation in this excerpt."),
    ("Vp5", "The character's achievements, capabilities or actions receive visible validation, retribution or "
            "appreciation from the community or social systems. If they go unrecognized or are punished, score low. "
            "N/A only if no community or social system appears in the excerpt."),
]

# Prompt v4 = v3 + regla de umbral en el paso 1.
PROMPT_PUNTUAR = """
### ROLE
You are acting as a literary analysis judge. You will read ONE excerpt of a literary work and answer a list of statements about how the work treats a specific character in that excerpt.

### CORE PRINCIPLES
1. Neutrality: Maintain strict neutrality regarding gender, age, social background, class, or culture.
2. No Moralism: Do not evaluate the work according to contemporary moral standards or external ideological classifications.
3. Strict Evidence: Analyze ONLY what the text shows and how it shows it.
4. Narrative filters: If the excerpt is narrated by a biased character or an unreliable narrator, look for the author's true stance in the 'cracks' of that same passage: dramatic irony, contradictions, the emotional weight of the descriptions, or events that contradict the narrator.

### SCORING RULES (MANDATORY)
Answer EVERY statement in the list below, using its label (S1, S2, ...). The statements are in random order and are independent of one another: judge each one on its own, and do not try to group them or keep your answers consistent across them.

For each statement, reason in TWO STEPS, in this order, and only then give the score:

STEP 1 · Material check. Begin your reasoning with exactly "Material: yes" or "Material: no", followed by one sentence naming what in the excerpt is (or is not) relevant to the statement. Use the statement's own N/A condition as the test: answer "no" only if that case applies (e.g. the character interacts with nobody, nobody pressures them, nothing changes). The ABSENCE of what the statement describes is still material: if other characters are present but nobody values the character, that is "Material: yes" (and a low score). If the character faces the situation and overcomes it, that is "Material: yes" (and a high score).
Threshold: the situation a statement describes counts as present even when it is mild, indirect or implicit (a disapproving look, a household rule, an unspoken expectation, a private conversation). Do NOT require an overt, dramatic, explicit or "formal" version of it; how strong it is belongs in the score, not in the material check. When in doubt whether the situation is present, answer "Material: yes" and score it.

STEP 2 · Score. If "Material: no", the score MUST be N/A. If "Material: yes", the score MUST be a number from 1 to 5, never N/A; explain it in 1-3 sentences connecting the text to the criterion.

Be equally careful with the last statement as with the first.

Assign exactly ONE value:
- 1: Strongly disagree
- 2: Disagree
- 3: Neutral
- 4: Agree
- 5: Totally Agree
- N/A: only when your Material check is "no".

**N/A rule (applies to every statement).** Score N/A only when the excerpt gives no material at all to judge the statement (e.g. the character is absent, or no other characters/social world appear). If the material is there but the statement is not fulfilled, score 1 or 2, never N/A. If the character faces the situation the statement describes and responds well, score high: overcoming the situation is never a reason for N/A. Each statement's "N/A only if..." clause names the specific case of no material for that statement.
**Base your answer ONLY on the excerpt provided. DO NOT IMPROVISE and do not use prior knowledge of the work.**

### STATEMENTS
{questions}

### INPUT
Character: {character}
Excerpt (id {unit_id}, narrative stage {stage}): \"\"\"
{excerpt}
\"\"\"
"""
