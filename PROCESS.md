# How I plan the next work (Scrum-style)

This is the plan for the next sprints of **digital-twin-pump-simulation**. It is a planned way of working for a one-person project, not a record of past team work.

## Working agreement
- Sprint: one week. Sprint goal: one backlog item finished, with its numbers in `reports/` and the README.
- Planning on Monday (pick items), review on Friday (open the project page, check the numbers), a short retrospective note in the commit log.
- Board: one GitHub issue per backlog item, columns To do, In progress, Done (GitHub Projects).
- Branches: `main` stays working; each item gets its own branch and a pull request.

## Definition of done
1. The script runs in a fresh virtual environment from `requirements.txt`.
2. Every number in the README and on the page matches `reports/`.
3. The project page opens without console errors and works on a phone.
4. No secrets or personal data in the files or history.
5. Limits are stated next to the result.

## Product backlog
| # | User story | Acceptance |
|---|---|---|
| 1 | As a reader, I want harder, overlapping faults, so that the task is not trivially separable. | Faults overlap in two sensors; a threshold rule no longer matches the forest. |
| 2 | As a reader, I want real pump data, so that the twin is checked against reality. | One open dataset evaluated; licence recorded. |
| 3 | As a reader, I want a working sequence model, so that the LSTM attempts end with a result. | LSTM trained properly on runs, compared with the forest. |
