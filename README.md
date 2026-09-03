# litfem

Funciones genéricas para el análisis literario PTM-V con LLM (LangChain + OpenAI,
salidas estructuradas con Pydantic).

La librería **no contiene prompts**: el template de instrucciones y las preguntas
de cada dimensión se definen en el notebook y se pasan como argumento, para poder
experimentar con ellos sin tocar el repo.

## Instalación (Colab)

```python
%pip install -q "git+https://github.com/USUARIO/REPO.git"
```

## Uso mínimo

```python
from litfem import configurar_api_key, analizar_personaje_completo, mostrar_resumen_global, PTMVResult

configurar_api_key()

resultados = analizar_personaje_completo(
    character="Alicia",
    fragment=texto_del_resumen,
    instructions=PTM_V_INSTRUCTIONS,      # definido en el notebook
    dimensiones=PTM_V_DIMENSIONES_POR_GRUPO,
    formato=PTMVResult,
    excel="resultados.xlsx",
)
mostrar_resumen_global(resultados)
```

`instructions` debe tener los placeholders `{dimensiones_texto}`, `{character}`,
`{dimension}` y `{fragment}`. `dimensiones` es un `dict {clave: texto de preguntas}`.

## API

| Función | Qué hace |
|---|---|
| `configurar_api_key()` | Carga `OPENAI_API_KEY` desde entorno, Colab Secrets, `.env` o `getpass` |
| `llm_function(prompt, esquema, ...)` | Llamada al LLM con salida forzada al esquema |
| `analizar_dimension_ptmv(...)` | Analiza una dimensión |
| `analizar_personaje_completo(...)` | Analiza todas las dimensiones en paralelo, muestra y exporta |
| `mostrar_resultados_bonitos(res)` | Tabla + explicaciones de N/A + promedios de una dimensión |
| `mostrar_resumen_global(resultados)` | Tabla final dimensión × momento |
| `exportar_resultados_excel(...)` | Excel de 4 hojas (Detalle, Promedios, Análisis, JSON crudo) |
| `PTMVResult` | Esquema Pydantic de salida (score restringido a 1-5 o N/A) |

## Publicar en GitHub

```bash
cd litfem-pkg
git init && git add . && git commit -m "litfem v0.1.0"
git remote add origin https://github.com/USUARIO/REPO.git
git push -u origin main
```
