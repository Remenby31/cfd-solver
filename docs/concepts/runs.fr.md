---
source_hash: "12a9f0032af3"
---

# Calculs

Un calcul est une exécution d'un cas sur un maillage. Il écrit un répertoire avec ses résultats et ce qui les identifie : le cas tel qu'exécuté, le hash du maillage et le commit du solveur. Le runtime du solveur `autan-run` et le backend `fake` du SDK écrivent la même structure, le [contrat de résultats](../contracts/results.md) ; `autan.Results` le lit.

## États d'un calcul

![États d'un calcul](../assets/diagrams/run-lifecycle.svg)

`autan plan` passe d'abord. S'il trouve une erreur, rien n'est écrit ; `autan-run` n'écrit rien non plus pour une physique qu'il ne calcule pas (`E0706`). Sinon le calcul démarre dans l'état `running` et se termine dans un seul état :

| État | Signification |
|---|---|
| `converged` | Chaque résidu a baissé du nombre d'ordres demandé. |
| `completed` | Le budget d'itérations ou de pas de temps s'est épuisé avant. |
| `stopped` | Vous, un signal ou une limite d'exécution l'avez arrêté. |
| `diverged` | La solution est devenue non finie ou non physique ; un diagnostic `D` nomme la cellule. |
| `failed` | Le backend a rencontré une erreur inattendue, `E0704`. |

## Le répertoire de calcul

| Fichier | Écrit |
|---|---|
| `run.json` | Au démarrage, puis à chaque changement d'état. |
| `case.toml` | Au démarrage : le cas tel qu'exécuté, toutes les valeurs par défaut remplies. |
| `mesh-info.json` | Au démarrage : le MeshInfo du maillage lu. |
| `history.jsonl` | Une ligne par itération enregistrée, pendant le calcul. |
| `solution.cgns` | À la fin, avant que l'état quitte `running` ; le schéma montre quels états finaux en ont une. |

Vous pouvez ouvrir un calcul en cours (`running`) : l'historique est lisible, la solution ne l'est pas.

## Comment un calcul est identifié

Chaque calcul enregistre le hash du maillage, le hash du cas, le commit du solveur et la précision. Leur combinaison, avec le backend et les périphériques, est l'`identity` du calcul ([Identity](../contracts/results.md#identity)). Avec `physics.reproducibility = "run"` ou `"ranks"`, deux calculs du solveur de même identité arrêtés à la même itération écrivent le même historique et les mêmes données de solution, au bit près. [run.json](../contracts/results.md#runjson) définit chaque champ.
