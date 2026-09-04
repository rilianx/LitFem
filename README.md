# litfem

Análisis literario PTM-V de personajes con LLM (LangChain + OpenAI, salidas estructuradas).

La librería **no contiene prompts ni preguntas**: viven en el notebook y se pasan como argumento.

## Un solo procedimiento

```
unidades  →  puntuar_unidades  →  consolidar
```

Una *unidad* es lo que el modelo ve en UNA llamada, etiquetada con su etapa narrativa (A/B/C).
En cada llamada viajan **todas las preguntas**, barajadas y con etiquetas neutras `S1…Sn`, sin
decir a qué dimensión pertenecen. Los tres modos solo difieren en cómo se construyen las unidades:

| Modo | Unidades | Llamadas |
|---|---|---|
| A · resumen | `unidades_resumen(texto)` — los momentos que el texto trae marcados | 3 |
| B · libro | `unidades_libro(texto)` — fragmentos ~2000 palabras, muestra 5-10-5 | 20 |
| C · evidencia → juez | `extraer_evidencia(...)` + `unidades_evidencia(df)` | 20 + 3 |

Los tres devuelven **el mismo formato largo**: una fila por respuesta con `iteracion, dimension,
momento, pregunta, score, score_txt, reasoning, fuente, etiqueta, posicion`.

```python
from litfem import (configurar_api_key, parsear_preguntas, unidades_libro,
                    puntuar_unidades, consolidar, mostrar_consolidado)

configurar_api_key()
PREGUNTAS = parsear_preguntas([("P1", "enunciado…"), ("P2", "…"), ("Vr3", "…")])

unidades = unidades_libro(texto_libro, max_words=2000, seleccion=(5, 10, 5))
largo = puntuar_unidades(unidades, "Alicia", PROMPT_PUNTUAR, PREGUNTAS, semilla=0)
mostrar_consolidado(consolidar(largo), "Libro por fragmentos")
```

`PROMPT_PUNTUAR` usa `{character} {stage} {unit_id} {excerpt} {questions}`; el extractor añade
`{max_citas}` y el juez `{fragment_ids}`. Las claves no usadas se ignoran.

## API

**Preguntas y unidades**

| Función | Qué hace |
|---|---|
| `parsear_preguntas(preguntas)` | Normaliza el banco; acepta lista de pares `("P1", "…")` (recomendado), lista de dicts o el dict por dimensión. La dimensión sale del prefijo del id |
| `barajar(preguntas, semilla)` / `formatear_preguntas(orden)` | Orden aleatorio reproducible y bloque para el prompt |
| `esquema_puntajes(n)` | Modelo Pydantic con S1..Sn obligatorios (Structured Output) |
| `unidades_resumen(texto, patron=)` | 3 unidades por los marcadores del resumen; si no los halla, corta 25/50/25 |
| `unidades_libro(texto, max_words, seleccion)` | Fragmentos + muestra + etapa |
| `unidades_evidencia(df, solo_verificadas=, excluir_fragmentos=)` | 1 unidad por etapa con la evidencia formateada |

**Núcleo y modo C**

| Función | Qué hace |
|---|---|
| `puntuar_unidades(unidades, personaje, prompt, preguntas, semilla=, iteracion=)` | Una llamada por unidad → formato largo |
| `iterar(fn, num_iteraciones)` | Repite cambiando semilla e iteración, y concatena |
| `extraer_evidencia(unidades, personaje, prompt, preguntas, max_citas=)` | Citas verbatim con polaridad, verificadas contra el texto |
| `resumen_evidencia(df)` / `mostrar_evidencia(df, dimension=, etapa=)` | Conteo y listado de citas |
| `agregar_jackknife(obs, error)` | Agregador con error jackknife para el leave-one-out |

**Consolidación y vistas**

| Función | Qué hace |
|---|---|
| `consolidar(largo, error=, agregador=, combinador=)` | (1) por dimensión×momento×pregunta, (2) media de preguntas por dimensión×momento |
| `agregar_celda` / `combinar_preguntas` | Agregador y combinador por defecto; reemplazables desde el notebook |
| `mostrar_consolidado(cons)` | Tablas `media ± error`, más n y N/A |
| `comparar_modos({...})` / `comparar_consolidados(a, b)` | Varios modos lado a lado / diferencia entre dos |
| `residuos_posicion(largo, control=)` | Resta lo esperado para esa misma pregunta: `"celda"` (pareado, misma pregunta y unidad, necesita iteraciones), `"aditivo"` (descuenta efecto de pregunta y de unidad) o `"auto"` |
| `efecto_posicion(largo, control=)` | Residuo medio por tramo + pendiente residuo ~ posición con su t **y su magnitud** (puntos de punta a punta, sd, % de variabilidad); también la tendencia de N/A y del largo del razonamiento |
| `tabla_cruda`, `mostrar_razonamientos`, `mostrar_na` | Vistas del detalle crudo |
| `acuerdo(referencia, surrogado, grafico=)` | ¿Sirve un modo como surrogado de otro? Sesgo, MAE, límites de acuerdo, z por celda, dónde se concentra el desacuerdo, calibración; con `grafico=True`, identidad + Bland-Altman |
| `exportar_consolidado_excel`, `exportar_largo_excel` | Excel de tablas y de observaciones |

**Validación del instrumento** (sobre datos ya obtenidos, sin gastar llamadas; requiere varias iteraciones)

| Función | Qué hace |
|---|---|
| `repetibilidad(largo)` | Test-retest: MAE entre corridas del mismo modo. Es el **techo** de acuerdo alcanzable contra cualquier otro modo |
| `fiabilidad_preguntas(largo)` | Por pregunta: tasa de N/A, ruido, y discriminación (¿separa unas unidades de otras?) |
| `curva_unidades(largo)` | Curva de saturación: cuántos fragmentos hacen falta para que la tabla deje de moverse |
| `comparar_agregadores({...}, largo)` | Evalúa agregadores por **señal/ruido**, no por el acuerdo con otro modo |

**Calibración del surrogado** (con un corpus de obras analizadas de las dos formas)

| Función | Qué hace |
|---|---|
| `pares_de_consolidados({obra: (cons_sur, cons_ref)})` | Tabla de pares (obra, dimensión, momento, surrogado, referencia) |
| `calibrar(pares)` | Ajusta desfase/recta, global o por dimensión, y los valida **dejando una obra fuera**: el MAE de validación es el error esperado en una obra nueva |
| `MODELOS` | Los modelos de calibración disponibles, para aplicar el elegido a una obra nueva |

## Publicar en GitHub

```bash
cd litfem-pkg
git init && git add . && git commit -m "litfem v0.10.0"
git remote add origin https://github.com/USUARIO/REPO.git
git push -u origin main
```
