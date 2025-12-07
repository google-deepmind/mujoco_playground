# Doku für Jaywalker-Mujoco-Playground Adaptation

## Entwicklungsumgebung einrichten/Sachen die wir auf dem Server machen mussten

 - [uv](https://docs.astral.sh/uv/getting-started/installation/) installieren
 - ssh-Key für den Zugriff auf Github (läuft aktuell auf Privataccount) eingerichtet
 - [Projekt-Repo](https://github.com/BigSmoke908/mujoco_playground) in das Homeverzeichnis gecloned: `git clone https://github.com/BigSmoke908/mujoco_playground`
 - `cd mujoco_playground`
 - `uv venv --python 3.11`, lokale von uv verwaltete Python Umgebung erstellen
 - `source .venv/bin/activate`, die Umgebung aktivieren
 - `uv pip install -U "jax[cuda12]"`
 - JAX/CUDA-Installation testen: `python -c "import jax; print(jax.default_backend())"`  (sollte `gpu` ausgeben)
 - `uv pip install -e ".[all]"`, alle anderen Dependencies für Mujoco-Playground installieren
 - komplette Installation verifizieren: `python -c "import mujoco_playground"`  (sollte ohne Fehler durchlaufen)


## Mujoco-Simulation öffnen
> Hierfür muss Mujoco installiert sein. Am einfachsten geht das, indem man die obige Installationanleitung für den Playground befolgt.

 - `cd mujoco_playground`, in den lokalen Projekt-Clon wechseln
 - `source .venv/bin/activate`, die lokale Python Umgebung aktivieren
 - `python`, Python-Interpreter starten
 - `import mujoco.viewer as m`, Mujoco Viewer Modul importieren
 - `m.launch()`  -> ein Fenster mit der Mujoco-Simulation öffnet sich
 - Nun kann man verschiedene Umgebungen laden in dem man diese ins Fenster zieht


## Training
> Über das Script [train_jax_ppo.py](./learning/train_jax_ppo.py) wird aktuell das Training ausgeführt

 - `cd mujoco_playground`  -> hier den lokalen Clon vom Repository wechseln
 - `source ./venv/activate`,  um die lokale Python Umgebung zu aktivieren (muss pro Terminal-Session nur einmal durchgeführt werden)
 - `python learning/train_jax_ppo.py --env_name=WolvesOPJoystickFlatTerrain`, Standardaufruf für das Training


### einige Optionale Trainingsparameter

> Parameter sind über `flags.DEFINE_...`-Aufrufe in [train_jax_ppo.py](./learning/train_jax_ppo.py) definiert.

 - `--help`, Liste mit allen Parametern + Erklärung ausgeben
 - `--domain_randomization=false`, Domain-Randomization togglen
 - `--num_timesteps=150000000`, wie viele Episoden(?) sollen trainiert werden -> grobes Maß für wie lange trainiert werden soll
 - `--load_checkpoint_path={path}`, Skript mit einem bestimmten Trainingsstand laden, um das Training fortzusetzen, `--num_timesteps`sollte dann aber neugesetzt werden
    - Beispiel: `--load_checkpoint_path logs/WolvesOPJoystickFlatTerrain-20251104-075648/checkpoints --num_timesteps=200000000`
 - `--play_only=true`, es wird kein Training ausgeführt sondern nur ein Video erstellt, vom letzten Trainingsstand (oder von einem bestimmten wenn `--load_checkpoint_path` gesetzt ist)
 - `--use_tb=true`, Loggingdirectory für Tensorboard erstellen
 - `--num_evals=10`, wie viele Zwischenstände sollen während dem Training gespeichert werden




