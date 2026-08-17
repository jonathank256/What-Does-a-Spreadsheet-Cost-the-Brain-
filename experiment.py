# =========================
# IMPORTS
# =========================
from psychopy import visual, core, event, gui
import csv
import random
import os
import json
import statistics
from datetime import datetime
import string

# =========================
# CONFIG
# =========================
N_ROWS = 16 # rows in stimuli tables
N_COLS = 16 # columns in stimuli tables
FIXATION_DURATION_SEC = 1.5 # fixation between trials
USE_TRIGGERS = False # False without hardware, True with
EYE_TRACKING = False # if eye-tracking
USE_EEG = False # if EEG
EEG_STREAM_NAME = "gNautilus" # default LSL stream. may need to update. "g.NEEDaccess client"
EEG_STREAM_TYPE = "EEG"
EEG_N_CHANNELS = 16 # may be 8 or 16 depending on strand use. defaulting to maximum
EEG_CONNECT_TIMEOUT_SEC = 10 # how long to wait for stream at startup
RUN_INSTRUCTION_PHASE = True # False to skip instructions/demos (e.g. for testing)

DATA_DIR = "psychopy_output"
STIM_DIR = os.path.join(DATA_DIR, "stimuli_csv") # saving stimuli tables
RESULTS_DIR = os.path.join(DATA_DIR, "results") # results location
# ensuring these directories exist
os.makedirs(STIM_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# =========================
# TRIAL SETTINGS
# =========================
TRIAL_DURATION_SEC = 30
N_TRIAL_REPEATS = 1
BREAK_DURATION_SEC = 5
FULLSCREEN = True
WINDOW_SIZE = [2560, 1600]

# =========================
# TESTING / ORDER CONTROL
# =========================
FORCE_FIRST_BLOCK_ENABLED = False # allows for control over starting trial, debug
FORCE_FIRST_BLOCK = ("spatial", "hard") # set which to force
RUN_ONLY_FORCED_BLOCK = False
FIXED_BLOCK_ORDER = False

# Spatial Settings
SPATIAL_EASY_TARGET_COLOR = "#0055CC"

# Visibility Settings
VISIBILITY_EASY_TARGET_COLOR     = "black"
VISIBILITY_EASY_DISTRACTOR_COLOR = "#8E8E8E"
VISIBILITY_HARD_TARGET_COLOR     = "#CECECE"
VISIBILITY_HARD_DISTRACTOR_COLOR = "#D8D8D8"

# =========================
# LSL TRIGGER STREAM (NIRSport 2 via USB)
# =========================
outlet = None
if USE_TRIGGERS:
    from pylsl import StreamInfo, StreamOutlet
    info = StreamInfo(
        name="PsychoPyTriggers",
        type="Markers",
        channel_count=1,
        nominal_srate=0,
        channel_format="int32",
        source_id="psychopy_exp"
    )
    outlet = StreamOutlet(info)
    core.wait(3)   # give Aurora time to discover the stream before first trigger

# =========================
# EEG (g.tec gNautilus, LSL)
# =========================
eeg_inlet = None

if USE_EEG:
    from pylsl import StreamInlet, resolve_byprop

    def connect_eeg_stream(timeout=EEG_CONNECT_TIMEOUT_SEC):
        """
        Discover gNautilus stream, return an inlet.
        g.NEEDaccess must be running AND streaming before this is called,
        returns None if not found before timeout.
        """
        print(f"[EEG] Searching for LSL stream '{EEG_STREAM_NAME}' (type='{EEG_STREAM_TYPE}')...")
        streams = resolve_byprop("name", EEG_STREAM_NAME, timeout=timeout)
        if not streams:
            print(f"[EEG] WARNING: No stream found within {timeout}s.")
            return None
        inlet = StreamInlet(streams[0], maxbuflen=30)
        info = inlet.info()
        print(
            f"[EEG] Connected: {info.name()} | "
            f"{info.channel_count()} ch @ {info.nominal_srate()} Hz | "
            f"source_id={info.source_id()}"
        )
        return inlet

    def check_eeg_stream_alive(inlet):
        """Non-blocking pull to verify samples are working."""
        if inlet is None:
            return False
        sample, _ = inlet.pull_sample(timeout=0.5)
        return sample is not None

# =========================
# Eye Tracking
# =========================
el = None  # this is assigned in main() after win is created

if EYE_TRACKING:
    import pylink
    from EyeLinkCoreGraphicsPsychoPy import EyeLinkCoreGraphicsPsychoPy

# =========================
# TRIGGER BASE CODES
# =========================
# Composite trigger = round_num * 10000 + response_count * 100 + base_code
#
# Onsets and timeouts use response_count = 0
# Examples:
# spatial hard onset round 1: 1*10000 + 0*100  + 12 = 10012
# spatial hard timeout  round 1: 1*10000 + 0*100  + 72 = 10072
#
# Responses use the nth correct response within the trial:
# computation easy response round 1, 23rd correct: 1*10000 + 23*100 + 51 = 12351
#
# Demo triggers use round_num = 0 and response_count = 0: equals base_code only.
# NOTE: All real trial codes are >= 10011, so demo codes (11-92) are unambiguous.
#
TRIGGER_CODES = {
    ("spatial",     "easy",  "onset"   ): 11,
    ("spatial",     "hard",  "onset"   ): 12,
    ("computation", "easy",  "onset"   ): 21,
    ("computation", "hard",  "onset"   ): 22,
    ("visibility",  "easy",  "onset"   ): 31,
    ("visibility",  "hard",  "onset"   ): 32,

    ("spatial",     "easy",  "response"): 41,
    ("spatial",     "hard",  "response"): 42,
    ("computation", "easy",  "response"): 51,
    ("computation", "hard",  "response"): 52,
    ("visibility",  "easy",  "response"): 61,
    ("visibility",  "hard",  "response"): 62,

    ("spatial",     "easy",  "timeout" ): 71,
    ("spatial",     "hard",  "timeout" ): 72,
    ("computation", "easy",  "timeout" ): 81,
    ("computation", "hard",  "timeout" ): 82,
    ("visibility",  "easy",  "timeout" ): 91,
    ("visibility",  "hard",  "timeout" ): 92,
}

# =========================
# DEMO INSTRUCTIONS
# =========================
DEMO_INSTRUCTIONS = {
    ("spatial", "easy"): (
        "A target number is shown at the top. It appears in a unique colour.\n"
        "Click it. A new target will appear immediately after.\n\n"
        "Press SPACE to continue."
    ),
    ("spatial", "hard"): (
        "A target number and color are shown at the top.\n"
        "Find and click it in the table.\n"
        "A new target will appear immediately after.\n\n"
        "Press SPACE to continue."
    ),
    ("computation", "easy"): (
        "You will be asked to sum three values from a specific column and row range.\n"
        "Type your answer and press ENTER to submit. BACKSPACE to correct.\n"
        "A new question appears after each correct answer.\n\n"
        "Press SPACE to continue."
    ),
    ("computation", "hard"): (
        "Same as before, but with larger numbers.\n"
        "Type your answer and press ENTER to submit. BACKSPACE to correct.\n\n"
        "Press SPACE to continue."
    ),
    ("visibility", "easy"): (
        "A target number is shown at the top. The table numbers are faded,\n"
        "but the target appears darker. Click it.\n"
        "A new target will appear immediately after.\n\n"
        "Press SPACE to continue."
    ),
    ("visibility", "hard"): (
        "Same as before, but the numbers are much harder to read.\n"
        "The target is a slightly different shade of grey than the rest.\n"
        "Find and click the cell containing the target.\n\n"
        "Press SPACE to continue."
    )
}

DEMO_ORDER = [
    ("spatial",     "easy"),
    ("spatial",     "hard"),
    ("computation", "easy"),
    ("computation", "hard"),
    ("visibility",  "easy"),
    ("visibility",  "hard"),
]

# =========================
# TRIGGER HELPERS
# =========================

def _make_trigger_code(base_code, round_num=0, response_count=0):
    return round_num * 10000 + response_count * 100 + base_code

def _fire_port(code):
    if not USE_TRIGGERS or outlet is None:
        return
    outlet.push_sample([code])

def send_trigger(base_code, round_num=0, response_count=0, label=""):
    """
    Compute and fire a composite integer trigger.
    Falls back to console print when USE_TRIGGERS is False for debugging.
    EyeLink receives a readable string message regardless.
    """
    code = _make_trigger_code(base_code, round_num, response_count)

    if not USE_TRIGGERS or outlet is None:
        print(f"[TRIGGER] code={code} (base={base_code} r={round_num} n={response_count}) {label} t={core.getTime():.4f}")
    else:
        _fire_port(code)

    if EYE_TRACKING and el is not None:
        el.sendMessage(f"TRIGGER {code} {label}")

# =========================
# EYETRACKING TRIGGER HELPERS
# =========================
def send_block_start(round_num):
    if EYE_TRACKING and el is not None:
        el.sendMessage(f"BLOCKID R{round_num}")
        el.sendMessage(f"BLOCK_START R{round_num}")

def send_block_end(round_num):
    if EYE_TRACKING and el is not None:
        el.sendMessage(f"BLOCK_END {round_num}")

def send_trial_start(trial_num, round_num, trial_in_round, task_type, difficulty):
    if EYE_TRACKING and el is not None:
        el.sendMessage(f"TRIALID T{trial_num}")
        el.sendMessage(f"!V TRIAL_VAR Trial {trial_num}")
        el.sendMessage(f"!V TRIAL_VAR Round {round_num}")
        el.sendMessage(f"!V TRIAL_VAR TrialInRound {trial_in_round}")
        el.sendMessage(f"!V TRIAL_VAR TaskType {task_type}")
        el.sendMessage(f"!V TRIAL_VAR Difficulty {difficulty}")

def send_trial_end():
    if EYE_TRACKING and el is not None:
        el.sendMessage("TRIAL_RESULT 0")

# =========================
# DATA GENERATION
# =========================
def make_headers(n_cols):
    return ["PID"] + [f"Feature_{i:02d}" for i in range(1, n_cols + 1)]

def generate_value_easy_spatial():
    return random.randint(10, 99)

def generate_value_easy_computation():
    return random.randint(0, 3)

def generate_value_hard_computation():
    tens = random.randint(1, 9)
    ones = random.randint(5, 9)
    return tens * 10 + ones

def generate_value_easy_visibility():
    return random.randint(10, 99)

def generate_value_hard_visibility():
    return random.randint(10, 99)

def get_all_trial_conditions():
    return [
        ("spatial", "easy"),
        ("spatial", "hard"),
        ("computation", "easy"),
        ("computation", "hard"),
        ("visibility", "easy"),
        ("visibility", "hard"),
    ]

def index_to_letter(i):
    letters = string.ascii_uppercase
    result = ""
    while True:
        result = letters[i % 26] + result
        i = i // 26 - 1
        if i < 0:
            break
    return result

def build_trial_sequence():
    base_conditions = get_all_trial_conditions()
    full_sequence = []
    previous_order = None

    for _ in range(N_TRIAL_REPEATS):
        current_order = base_conditions[:]
        random.shuffle(current_order)

        while previous_order is not None and current_order == previous_order:
            random.shuffle(current_order)

        full_sequence.extend(current_order)
        previous_order = current_order[:]

    return full_sequence

def generate_dataset(task_type, difficulty, n_rows=N_ROWS, n_cols=N_COLS):
    data = []

    for _ in range(n_rows):
        row = []
        for _ in range(n_cols):
            if task_type == "spatial" and difficulty == "easy":
                val = generate_value_easy_spatial()
            elif task_type == "spatial" and difficulty == "hard":
                val = generate_value_easy_spatial()
            elif task_type == "computation" and difficulty == "easy":
                val = generate_value_easy_computation()
            elif task_type == "computation" and difficulty == "hard":
                val = generate_value_hard_computation()
            elif task_type == "visibility" and difficulty == "easy":
                val = generate_value_easy_visibility()
            elif task_type == "visibility" and difficulty == "hard":
                val = generate_value_hard_visibility()
            else:
                raise ValueError("Invalid task_type/difficulty combination.")
            row.append(val)
        data.append(row)

    return data

def save_dataset_csv(filename, data):
    headers = make_headers(len(data[0]))
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for i, row in enumerate(data):
            writer.writerow([f"{i+1:02d}"] + row)

# =========================
# TRIAL TARGETS
# =========================
def make_spatial_target(data):
    r = random.randrange(len(data))
    c = random.randrange(len(data[0]))
    return {
        "row": r,
        "col": c,
        "value": data[r][c]
    }

def make_computation_target(data):
    col = random.randrange(len(data[0]))
    start_row = random.randrange(len(data) - 2)
    rows = [start_row, start_row + 1, start_row + 2]
    values = [data[r][col] for r in rows]
    correct_sum = sum(values)

    return {
        "col": col,
        "rows": rows,
        "values": values,
        "correct_sum": correct_sum
    }

# =========================
# TARGET UNIQUENESS
# =========================
def ensure_unique_target(data, target, task_type, difficulty, cell_lookup=None):
    # handled separately due to colour
    if task_type == "spatial" and difficulty == "easy":
        return
    if task_type == "spatial" and difficulty == "hard":
        return

    tr, tc = target["row"], target["col"]
    tval = target["value"]
    n_rows = len(data)
    n_cols = len(data[0])

    for r in range(n_rows):
        for c in range(n_cols):
            if r == tr and c == tc:
                continue
            if data[r][c] == tval:
                if task_type == "spatial" and difficulty == "hard":
                    gen = generate_value_easy_spatial
                elif task_type == "visibility" and difficulty == "easy":
                    gen = generate_value_easy_visibility
                else:
                    gen = generate_value_hard_visibility

                replacement = tval
                while replacement == tval:
                    replacement = gen()

                data[r][c] = replacement

                if cell_lookup is not None and (r, c) in cell_lookup:
                    cell_lookup[(r, c)]["text"].text = str(replacement)
                    cell_lookup[(r, c)]["value"]     = replacement

# =========================
# SPATIAL HARD HELPERS
# =========================

def make_spatial_hard_color_grid(n_rows, n_cols):
    """
    Random 50/50 assignment of colors, blue and black.
    Grid is fixed for duration of trial.
    """
    total = n_rows * n_cols
    half = total // 2
    flat = [SPATIAL_EASY_TARGET_COLOR] * half + ["black"] * (total - half)
    random.shuffle(flat)
    return [flat[r * n_cols: (r + 1) * n_cols] for r in range(n_rows)]

def make_spatial_hard_target(data, color_grid):
    """
    Picks a random cell; its color from color grid is part
    of the target definition, i.e. must pick right number and color.
    """
    r = random.randrange(len(data))
    c = random.randrange(len(data[0]))
    return {"row": r, "col": c, "value": data[r][c], "color": color_grid[r][c]}

def ensure_unique_spatial_hard_target(data, target, color_grid, cell_lookup=None):
    """
    Guarantee the target value is unique among the cells with the same target color.
    Duplicates are allowed in other color.
    """
    tr, tc = target["row"], target["col"]
    tval = target["value"]
    tcolor = target["color"]

    for r in range(len(data)):
        for c in range(len(data[0])):
            if r == tr and c == tc:
                continue
            if color_grid[r][c] == tcolor and data[r][c] == tval:
                replacement = tval
                while replacement == tval:
                    replacement = generate_value_easy_spatial()
                data[r][c] = replacement
                if cell_lookup is not None and (r, c) in cell_lookup:
                    cell_lookup[(r, c)]["text"].text = str(replacement)
                    cell_lookup[(r, c)]["value"] = replacement

# =========================
# DISPLAY HELPERS
# =========================
def format_time(t):
    mins = int(t) // 60
    secs = int(t) % 60
    return f"{mins:02d}:{secs:02d}"

def show_message(win, text="", wait_for_space=True, max_wait=None, height=0.05):
    msg = visual.TextStim(
        win,
        text=text,
        color="black",
        height=height,
        wrapWidth=1.6
    )

    if text.strip() == "":
        win.flip()
    else:
        msg.draw()
        win.flip()

    if wait_for_space:
        event.clearEvents()

        if max_wait is None:
            keys = event.waitKeys(keyList=["space", "escape"])
        else:
            keys = event.waitKeys(maxWait=max_wait, keyList=["space", "escape"])

        if keys and "escape" in keys:
            core.quit()

def show_fixation(win, seconds=FIXATION_DURATION_SEC):
    fixation = visual.TextStim(
        win,
        text="+",
        color="black",
        height=0.12
    )

    timer = core.Clock()
    event.clearEvents()

    while timer.getTime() < seconds:
        fixation.draw()
        win.flip()

        keys = event.getKeys(["escape"])
        if "escape" in keys:
            core.quit()

def break_screen(win, block_name, seconds=BREAK_DURATION_SEC):
    msg = visual.TextStim(win, color="black", height=0.045, wrapWidth=1.6)
    timer = core.Clock()
    event.clearEvents()

    while timer.getTime() < seconds:
        remaining = max(0, seconds - timer.getTime())
        msg.text = (
            f"Take a short break.\n\n"
            f"Please wait {int(remaining)} more seconds."
        )
        msg.draw()
        win.flip()

        keys = event.getKeys(["escape"])
        if "escape" in keys:
            core.quit()

def build_table_stimuli(win, data, task_type=None, difficulty=None, cell_colors=None):
    n_rows = len(data)
    n_cols = len(data[0])

    left = -0.92
    right = 0.92
    top = 0.55
    bottom = -0.82

    row_label_width = 0.08
    col_header_height = 0.08

    table_left = left + row_label_width
    table_top = top - col_header_height

    table_width = right - table_left
    table_height = table_top - bottom

    cell_w = table_width / n_cols
    cell_h = table_height / n_rows

    uniform_text_color = "black"
    cell_fill_color = None

    if task_type == "visibility" and difficulty == "easy":
        uniform_text_color = VISIBILITY_EASY_DISTRACTOR_COLOR
        cell_fill_color = "#F5F5F5"
    elif task_type == "visibility" and difficulty == "hard":
        uniform_text_color = VISIBILITY_HARD_DISTRACTOR_COLOR
        cell_fill_color = "#F5F5F5"

    if task_type == "visibility":
        line_color = "#D0D0D0"
    else:
        line_color = "black"

    static_stims    = []
    clickable_cells = []

    for c in range(n_cols):
        x = table_left + cell_w * (c + 0.5)
        y = top - col_header_height / 2
        static_stims.append(
            visual.TextStim(win, text=index_to_letter(c), pos=(x, y),
                            color="red", height=min(0.028, cell_h * 0.55))
        )

    for r in range(n_rows):
        x = left + row_label_width / 2
        y = table_top - cell_h * (r + 0.5)
        static_stims.append(
            visual.TextStim(win, text=index_to_letter(r), pos=(x, y),
                            color="red", height=min(0.028, cell_h * 0.55))
        )

    for c in range(n_cols + 1):
        x = table_left + c * cell_w
        static_stims.append(
            visual.Line(win, start=(x, bottom), end=(x, table_top), lineColor=line_color)
        )

    for r in range(n_rows + 1):
        y = table_top - r * cell_h
        static_stims.append(
            visual.Line(win, start=(table_left, y), end=(right, y), lineColor=line_color)
        )

    for r in range(n_rows):
        for c in range(n_cols):
            x = table_left + cell_w * (c + 0.5)
            y = table_top - cell_h * (r + 0.5)

            txt_color = (
                cell_colors[r][c]
                if cell_colors is not None
                else uniform_text_color
            )

            rect = visual.Rect(
                win, width=cell_w, height=cell_h, pos=(x, y),
                lineColor=None, fillColor=cell_fill_color
            )

            txt = visual.TextStim(
                win, text=str(data[r][c]), pos=(x, y),
                color=txt_color, height=min(0.045, cell_h * 0.75)
            )

            clickable_cells.append({
                "row":       r,
                "col":       c,
                "row_label": index_to_letter(r),
                "col_label": index_to_letter(c),
                "value":     data[r][c],
                "color":     txt_color,
                "rect":      rect,
                "text":      txt
            })

    return static_stims, clickable_cells

def _make_even(v):
    if v % 2 != 0:
        v = v + 1 if v < 99 else v - 1
    return max(10, min(98, v))

def _make_odd(v):
    if v % 2 == 0:
        v = v + 1 if v < 99 else v - 1
    return max(11, min(99, v))

def assign_spatial_features(data, target):
    tr, tc = target["row"], target["col"]

    if target.get("difficulty") == "easy":
        cell_colors = [
            [SPATIAL_EASY_TARGET_COLOR if (r == tr and c == tc) else "black"
             for c in range(len(data[0]))]
            for r in range(len(data))
        ]
        return cell_colors, data

    return None, data

def reassign_target_feature(data, cell_lookup, old_target, new_target, difficulty):
    if difficulty == "easy":
        otr, otc = old_target["row"], old_target["col"]
        ntr, ntc = new_target["row"], new_target["col"]

        data[otr][otc] = _make_even(data[otr][otc])
        cell_lookup[(otr, otc)]["value"] = data[otr][otc]
        cell_lookup[(otr, otc)]["text"].text = str(data[otr][otc])
        cell_lookup[(otr, otc)]["text"].color = "black"

        data[ntr][ntc] = _make_odd(data[ntr][ntc])
        new_target["value"] = data[ntr][ntc]
        cell_lookup[(ntr, ntc)]["value"] = data[ntr][ntc]
        cell_lookup[(ntr, ntc)]["text"].text = str(data[ntr][ntc])
        cell_lookup[(ntr, ntc)]["text"].color = SPATIAL_EASY_TARGET_COLOR

    return new_target

def draw_table_stimuli(static_stims, clickable_cells):
    for stim in static_stims:
        stim.draw()
    for cell in clickable_cells:
        cell["rect"].draw()
        cell["text"].draw()

def point_in_cell(mouse_pos, cell, padding=0.005):
    x, y = mouse_pos
    cx, cy = cell["rect"].pos
    hw = cell["rect"].width / 2 + padding
    hh = cell["rect"].height / 2 + padding
    return (cx - hw <= x <= cx + hw) and (cy - hh <= y <= cy + hh)

# =========================
# DEMO TRIALS
# =========================

def run_demo_click_trial(win, mouse, task_type, difficulty):
    data = generate_dataset(task_type, difficulty)
    target = make_spatial_target(data)

    cell_colors = None
    if task_type == "spatial":
        target["difficulty"] = difficulty
        if difficulty == "easy":
            cell_colors, data = assign_spatial_features(data, target)
        else:
            color_grid = make_spatial_hard_color_grid(len(data), len(data[0]))
            target = make_spatial_hard_target(data, color_grid)
            ensure_unique_spatial_hard_target(data, target, color_grid)
            cell_colors = color_grid
    elif task_type == "visibility" and difficulty == "easy":
        tr, tc = target["row"], target["col"]
        cell_colors = [
            [VISIBILITY_EASY_TARGET_COLOR if (r == tr and c == tc)
             else VISIBILITY_EASY_DISTRACTOR_COLOR
             for c in range(len(data[0]))]
            for r in range(len(data))
        ]
    elif task_type == "visibility" and difficulty == "hard":
        tr, tc = target["row"], target["col"]
        cell_colors = [
            [VISIBILITY_HARD_TARGET_COLOR if (r == tr and c == tc)
             else VISIBILITY_HARD_DISTRACTOR_COLOR
             for c in range(len(data[0]))]
            for r in range(len(data))
        ]

    ensure_unique_target(data, target, task_type, difficulty)

    static_stims, clickable_cells = build_table_stimuli(
        win, data, task_type=task_type, difficulty=difficulty, cell_colors=cell_colors
    )

    cell_lookup = {(cell["row"], cell["col"]): cell for cell in clickable_cells}

    if task_type == "spatial" and difficulty == "easy":
        search_desc = f"Find and click the unique color number: {target['value']}."
    elif task_type == "spatial" and difficulty == "hard":
        color_name = "blue" if target["color"] == SPATIAL_EASY_TARGET_COLOR else "black"
        search_desc = f"Find and click the {color_name} cell containing: {target['value']}."
    else:
        search_desc = f"Find and click the cell containing: {target['value']}."

    demo_banner = visual.TextStim(
        win, text="[ DEMO -- this trial does not count ]",
        pos=(0, 0.92), color="#8B0000", height=0.038, bold=True
    )

    instr_stim = visual.TextStim(
        win, text=search_desc, pos=(0, 0.80),
        color="black", height=0.04, wrapWidth=1.8
    )

    hint_stim = visual.TextStim(
        win, text="No time limit. Click the correct cell to continue.",
        pos=(0, -0.90), color="#555555", height=0.034, wrapWidth=1.8
    )

    mouse.setVisible(True)
    event.clearEvents()

    demo_banner.draw()
    instr_stim.draw()
    hint_stim.draw()
    draw_table_stimuli(static_stims, clickable_cells)
    win.flip()

    mouse.clickReset()

    send_trigger(TRIGGER_CODES[(task_type, difficulty, "onset")],
                 label=f"DEMO_{task_type}_{difficulty}_onset")

    while True:
        if task_type == "spatial" and difficulty == "easy":
            instr_stim.text = f"Find and click the unique color number: {target['value']}."
        elif task_type == "spatial" and difficulty == "hard":
            color_name = "blue" if target["color"] == SPATIAL_EASY_TARGET_COLOR else "black"
            instr_stim.text = f"Find and click the {color_name} cell containing: {target['value']}."
        else:
            instr_stim.text = f"Find and click the cell containing: {target['value']}."

        demo_banner.draw()
        instr_stim.draw()
        hint_stim.draw()
        draw_table_stimuli(static_stims, clickable_cells)
        win.flip()

        keys = event.getKeys(["escape"])
        if "escape" in keys:
            core.quit()

        buttons, times = mouse.getPressed(getTime=True)
        if buttons[0] and times[0] > 0:
            mouse.clickReset()
            for cell in clickable_cells:
                if point_in_cell(mouse.getPos(), cell):
                    if task_type == "spatial" and difficulty == "hard":
                        is_correct = (
                            cell["value"] == target["value"] and
                            cell["color"] == target["color"]
                        )
                    else:
                        is_correct = cell["value"] == target["value"]

                    if is_correct:
                        send_trigger(TRIGGER_CODES[(task_type, difficulty, "response")],
                                     label=f"DEMO_{task_type}_{difficulty}_response")
                        _show_demo_success(win, demo_banner)
                        return
                    break

        core.wait(0.01)

def run_demo_computation_trial(win, difficulty):
    data = generate_dataset("computation", difficulty)
    target = make_computation_target(data)

    static_stims, clickable_cells = build_table_stimuli(
        win, data, task_type="computation", difficulty=difficulty
    )

    demo_banner = visual.TextStim(
        win, text="[ DEMO -- this trial does not count ]",
        pos=(0, 0.92), color="#8B0000", height=0.038, bold=True
    )

    prompt_stim = visual.TextStim(
        win, text="", pos=(0, 0.81), color="black", height=0.038, wrapWidth=1.8
    )
    hint_stim = visual.TextStim(
        win, text="No time limit. Enter the correct sum to continue.",
        pos=(0.45, -0.90), color="#555555", height=0.034, wrapWidth=1.8
    )
    answer_label = visual.TextStim(
        win, text="Answer:", pos=(-0.40, -0.90), color="red",
        height=0.045, anchorHoriz="center"
    )
    answer_box = visual.Rect(
        win, width=0.42, height=0.08, pos=(0.00, -0.90),
        lineColor="black", fillColor=None, lineWidth=2
    )
    answer_text_stim = visual.TextStim(
        win, text="", pos=(0.00, -0.90), color="black", height=0.045
    )
    feedback_stim = visual.TextStim(
        win, text="", pos=(0.00, -0.90), color="red", height=0.04
    )

    event.clearEvents()
    answer_text = ""
    feedback_text = ""
    feedback_until = None

    def _build_prompt():
        return (
            f"Type the sum of Column {index_to_letter(target['col'])} "
            f"for Rows {index_to_letter(target['rows'][0])}, "
            f"{index_to_letter(target['rows'][1])}, and "
            f"{index_to_letter(target['rows'][2])}.\n"
            f"Press RETURN to submit  |  BACKSPACE to delete"
        )

    prompt_stim.text = _build_prompt()
    demo_banner.draw()
    prompt_stim.draw()
    hint_stim.draw()
    draw_table_stimuli(static_stims, clickable_cells)
    answer_label.draw()
    answer_box.draw()
    answer_text_stim.draw()
    win.flip()

    send_trigger(TRIGGER_CODES[("computation", difficulty, "onset")],
                 label=f"DEMO_computation_{difficulty}_onset")

    while True:
        now = core.getTime()

        if feedback_until is not None and now >= feedback_until:
            feedback_text = ""
            feedback_until = None
            answer_box.lineColor = "black"
            answer_box.fillColor = None

        prompt_stim.text = _build_prompt()

        demo_banner.draw()
        prompt_stim.draw()
        hint_stim.draw()
        draw_table_stimuli(static_stims, clickable_cells)
        answer_label.draw()
        answer_box.draw()
        answer_text_stim.text = answer_text
        answer_text_stim.draw()
        if feedback_text:
            feedback_stim.text = feedback_text
            feedback_stim.draw()
        win.flip()

        keys = event.getKeys(keyList=[
            "escape", "return", "num_enter", "backspace",
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"
        ])

        for key in keys:
            if key == "escape":
                core.quit()
            elif key in "0123456789":
                answer_text += key
            elif key == "backspace":
                answer_text = answer_text[:-1]
            elif key in ["return", "num_enter"] and answer_text != "":
                response_num = int(answer_text)
                if response_num == target["correct_sum"]:
                    send_trigger(TRIGGER_CODES[("computation", difficulty, "response")],
                                 label=f"DEMO_computation_{difficulty}_response")
                    _show_demo_success(win, demo_banner)
                    return
                else:
                    feedback_text = "Incorrect -- try again."
                    feedback_stim.color = "red"
                    answer_box.lineColor = "red"
                    answer_box.fillColor = "#330000"
                    feedback_until = core.getTime() + 1.2
                    answer_text = ""

        core.wait(0.01)

def _show_demo_success(win, demo_banner):
    success = visual.TextStim(
        win, text="Well done!\n\nGet ready for the next task type.",
        pos=(0, 0), color="#006600", height=0.065, wrapWidth=1.6
    )
    demo_banner.draw()
    success.draw()
    win.flip()
    core.wait(1.8)

# =========================
# INSTRUCTION + DEMO PHASE
# =========================

def run_instruction_and_demo_phase(win, mouse):
    phase_intro = (
        "Before the real experiment begins, you will be guided through\n"
        "each task type one at a time.\n\n"
        "For each task you will:\n"
        "  1. Read a short description of what to do\n"
        "  2. Complete one practice trial at your own pace\n\n"
        "Practice trials send the same signals as real trials\n"
        "but are NOT included in your results.\n\n"
        "Press SPACE to start the practice."
    )
    show_message(win, phase_intro, wait_for_space=True, height=0.048)

    for task_type, difficulty in DEMO_ORDER:
        instr_text = DEMO_INSTRUCTIONS[(task_type, difficulty)]
        show_message(win, instr_text, wait_for_space=True, height=0.046)
        show_fixation(win)

        if task_type in ("spatial", "visibility"):
            run_demo_click_trial(win, mouse, task_type, difficulty)
        elif task_type == "computation":
            run_demo_computation_trial(win, difficulty)

    ready_text = (
        "Great work -- you have completed all the practice tasks!\n\n"
        "The real experiment will now begin.\n"
        f"You will complete {N_TRIAL_REPEATS} rounds of {len(get_all_trial_conditions())} trials each.\n"
        f"Each trial lasts up to {TRIAL_DURATION_SEC} seconds.\n\n"
        "Press SPACE when you are ready to start."
    )
    show_message(win, ready_text, wait_for_space=True, height=0.048)


# =========================
# RESPONSE TASKS
# =========================

def run_click_trial(win, mouse, trial_num, round_num, trial_in_round,
                    task_type, block_name, participant_id, response_writer):
    """
    Runs one click-based trial (spatial or visibility).

    Returns a trial-level summary dict.
    Writes one row per click event to response_writer.
    """
    data = generate_dataset(task_type, block_name)
    target = make_spatial_target(data)

    color_grid = None
    cell_colors = None

    if task_type == "spatial":
        target["difficulty"] = block_name
        if block_name == "easy":
            cell_colors, data = assign_spatial_features(data, target)
        else:
            color_grid = make_spatial_hard_color_grid(len(data), len(data[0]))
            target = make_spatial_hard_target(data, color_grid)
            ensure_unique_spatial_hard_target(data, target, color_grid)
            cell_colors = color_grid
    elif task_type == "visibility" and block_name == "easy":
        tr, tc = target["row"], target["col"]
        cell_colors = [
            [VISIBILITY_EASY_TARGET_COLOR if (r == tr and c == tc)
             else VISIBILITY_EASY_DISTRACTOR_COLOR
             for c in range(len(data[0]))]
            for r in range(len(data))
        ]
    elif task_type == "visibility" and block_name == "hard":
        tr, tc = target["row"], target["col"]
        cell_colors = [
            [VISIBILITY_HARD_TARGET_COLOR if (r == tr and c == tc)
             else VISIBILITY_HARD_DISTRACTOR_COLOR
             for c in range(len(data[0]))]
            for r in range(len(data))
        ]
    if not (task_type == "spatial" and block_name == "hard"):
        ensure_unique_target(data, target, task_type, block_name)

    # Snapshot the initial target before any reassignment
    initial_target = {
        "row": index_to_letter(target["row"]),
        "col": index_to_letter(target["col"]),
        "value": target["value"],
        "color": target.get("color", ""),
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    csv_name = f"{participant_id}_{block_name}_{task_type}_trial{trial_num}_{timestamp}.csv"
    csv_path = os.path.join(STIM_DIR, csv_name)
    save_dataset_csv(csv_path, data)

    static_stims, clickable_cells = build_table_stimuli(
        win, data, task_type=task_type, difficulty=block_name, cell_colors=cell_colors
    )

    cell_lookup = {(cell["row"], cell["col"]): cell for cell in clickable_cells}

    if task_type == "spatial" and block_name == "easy":
        search_desc = f"Find and click the unique color number: {target['value']}."
    elif task_type == "spatial" and block_name == "hard":
        color_name = "blue" if target["color"] == SPATIAL_EASY_TARGET_COLOR else "black"
        search_desc = f"Find and click the {color_name} cell containing: {target['value']}."
    else:
        search_desc = f"Find and click the cell containing: {target['value']}."

    instr_stim = visual.TextStim(
        win, text=search_desc, pos=(0, 0.80),
        color="black", height=0.04, wrapWidth=1.8
    )

    timer_stim = visual.TextStim(
        win, text="", pos=(0.78, 0.92), color="black", height=0.04
    )

    mouse.setVisible(True)
    event.clearEvents()

    correct_count  = 0
    total_clicks = 0 # all clicks (correct + incorrect)
    error_count  = 0 # incorrect clicks
    correct_rts = [] # RT of each correct response (s)
    # target_sequence: list of {target_num, row, col, value, color} for each target presented
    target_sequence = [{
        "target_num": 1,
        "row":   initial_target["row"],
        "col":   initial_target["col"],
        "value": initial_target["value"],
        "color": initial_target["color"],
    }]
    target_num = 1 # 1-indexed count of targets presented this trial
    response_count = 0 # correct responses (for trigger coding)
    last_trigger_response = ""

    base_onset = TRIGGER_CODES[(task_type, block_name, "onset")]
    base_response = TRIGGER_CODES[(task_type, block_name, "response")]
    base_timeout  = TRIGGER_CODES[(task_type, block_name, "timeout")]

    onset_code = _make_trigger_code(base_onset,   round_num)
    timeout_code = _make_trigger_code(base_timeout, round_num)

    instr_stim.draw()
    timer_stim.draw()
    draw_table_stimuli(static_stims, clickable_cells)
    win.flip()

    mouse.clickReset()

    trial_clock = core.Clock()
    send_trigger(base_onset, round_num=round_num,
                 label=f"{task_type}_{block_name}_onset_r{round_num}")

    while trial_clock.getTime() < TRIAL_DURATION_SEC:
        time_left = max(0, TRIAL_DURATION_SEC - trial_clock.getTime())
        timer_stim.text = f"Time left: {format_time(time_left)}"
        timer_stim.color = "red" if time_left < 5 else "black"

        if task_type == "spatial" and block_name == "easy":
            instr_stim.text = f"Find and click the unique color number: {target['value']}."
        elif task_type == "spatial" and block_name == "hard":
            color_name = "blue" if target["color"] == SPATIAL_EASY_TARGET_COLOR else "black"
            instr_stim.text = f"Find and click the {color_name} cell containing: {target['value']}."
        else:
            instr_stim.text = f"Find and click the cell containing: {target['value']}."

        instr_stim.draw()
        timer_stim.draw()
        draw_table_stimuli(static_stims, clickable_cells)
        win.flip()

        keys = event.getKeys(["escape"])
        if "escape" in keys:
            core.quit()

        buttons, times = mouse.getPressed(getTime=True)
        if buttons[0] and times[0] > 0:
            mouse.clickReset()
            click_t = trial_clock.getTime()
            for cell in clickable_cells:
                if point_in_cell(mouse.getPos(), cell):
                    total_clicks += 1

                    if task_type == "spatial" and block_name == "hard":
                        is_correct = (
                            cell["value"] == target["value"] and
                            cell["color"] == target["color"]
                        )
                    else:
                        is_correct = cell["value"] == target["value"]

                    if not is_correct:
                        error_count += 1

                    # ── per-response row ──────────────────────────────────
                    response_writer.writerow({
                        "participant_id": participant_id,
                        "trial_num": trial_num,
                        "round_num": round_num,
                        "trial_in_round": trial_in_round,
                        "task_type": task_type,
                        "difficulty": block_name,
                        "response_type": "click",
                        "target_num": target_num,
                        "target_row": index_to_letter(target["row"]),
                        "target_col": index_to_letter(target["col"]),
                        "target_value": target["value"],
                        "target_color": target.get("color", ""),
                        "response_row": cell["row_label"],
                        "response_col": cell["col_label"],
                        "response_value": cell["value"],
                        "response_color": cell["color"],
                        "correct": int(is_correct),
                        "rt_s": round(click_t, 4),
                        "submitted_answer": "",
                        "correct_answer": "",
                        "trigger_code": "",
                    })

                    if is_correct:
                        correct_count  += 1
                        response_count += 1
                        correct_rts.append(round(click_t, 4))

                        trig_code = _make_trigger_code(base_response, round_num, response_count)
                        last_trigger_response = trig_code
                        send_trigger(base_response,
                                     round_num=round_num,
                                     response_count=response_count,
                                     label=f"{task_type}_{block_name}_response_r{round_num}_n{response_count}")

                        old_target = target

                        if task_type == "spatial":
                            if block_name == "easy":
                                target = make_spatial_target(data)
                                target["difficulty"] = block_name
                                target = reassign_target_feature(
                                    data, cell_lookup, old_target, target, block_name
                                )
                            else:
                                target = make_spatial_hard_target(data, color_grid)
                                ensure_unique_spatial_hard_target(
                                    data, target, color_grid, cell_lookup
                                )
                        elif task_type == "visibility" and block_name == "easy":
                            target = make_spatial_target(data)
                            otr, otc = old_target["row"], old_target["col"]
                            ntr, ntc = target["row"], target["col"]
                            cell_lookup[(otr, otc)]["text"].color = VISIBILITY_EASY_DISTRACTOR_COLOR
                            cell_lookup[(ntr, ntc)]["text"].color = VISIBILITY_EASY_TARGET_COLOR
                            ensure_unique_target(data, target, task_type, block_name, cell_lookup)
                        elif task_type == "visibility" and block_name == "hard":
                            target = make_spatial_target(data)
                            otr, otc = old_target["row"], old_target["col"]
                            ntr, ntc = target["row"], target["col"]
                            cell_lookup[(otr, otc)]["text"].color = VISIBILITY_HARD_DISTRACTOR_COLOR
                            cell_lookup[(ntr, ntc)]["text"].color = VISIBILITY_HARD_TARGET_COLOR
                            ensure_unique_target(data, target, task_type, block_name, cell_lookup)

                        # Record the new target that was just presented
                        target_num += 1
                        target_sequence.append({
                            "target_num": target_num,
                            "row":   index_to_letter(target["row"]),
                            "col":   index_to_letter(target["col"]),
                            "value": target["value"],
                            "color": target.get("color", ""),
                        })

                    break

        core.wait(0.01)

    trial_duration = trial_clock.getTime()
    send_trigger(base_timeout, round_num=round_num,
                 label=f"{task_type}_{block_name}_timeout_r{round_num}")

    # Summary stats
    accuracy = (correct_count / total_clicks) if total_clicks > 0 else ""
    mean_rt = round(statistics.mean(correct_rts),   4) if correct_rts else ""
    median_rt = round(statistics.median(correct_rts), 4) if correct_rts else ""
    min_rt = round(min(correct_rts), 4) if correct_rts else ""
    max_rt = round(max(correct_rts), 4) if correct_rts else ""
    # Inter-response intervals between consecutive correct clicks
    iris = [round(correct_rts[i] - correct_rts[i-1], 4)
            for i in range(1, len(correct_rts))] if len(correct_rts) > 1 else []
    mean_iri = round(statistics.mean(iris), 4) if iris else ""

    return {
        # identity 
        "participant_id": participant_id,
        "trial_num": trial_num,
        "round_num": round_num,
        "trial_in_round": trial_in_round,
        "task_type": task_type,
        "difficulty": block_name,
        # stimulus
        "stimulus_csv": csv_path,
        # first (initial) target 
        "first_target_row": initial_target["row"],
        "first_target_col": initial_target["col"],
        "first_target_value": initial_target["value"],
        "first_target_color": initial_target["color"],
        # performance summary 
        "correct_count": correct_count,
        "total_clicks": total_clicks,
        "error_count": error_count,
        "accuracy": round(accuracy, 4) if accuracy != "" else "",
        "targets_presented": target_num,
        # RT summary (correct responses only, seconds)
        "rt_first_correct_s": correct_rts[0] if correct_rts else "",
        "rt_mean_s": mean_rt,
        "rt_median_s": median_rt,
        "rt_min_s": min_rt,
        "rt_max_s": max_rt,
        "mean_iri_s": mean_iri,
        # trial timing 
        "trial_duration_s": round(trial_duration, 4),
        # trigger codes 
        "trigger_onset": onset_code,
        "trigger_last_response":last_trigger_response,
        "trigger_timeout": timeout_code,
        # full sequences (compact JSON for archival) 
        "target_sequence_json": json.dumps(target_sequence),
        "correct_rts_json": json.dumps(correct_rts),
        # computation-only fields (empty for click trials) 
        "first_target_rows": "",
        "first_target_values": "",
        "correct_sum": "",
        "total_submissions": "",
        "incorrect_submissions":"",
    }


def run_computation_trial(win, trial_num, round_num, trial_in_round,
                          block_name, participant_id, response_writer):
    """
    Runs one computation trial.

    Returns a trial-level summary dict.
    Writes one row per submission to response_writer.
    """
    data = generate_dataset("computation", block_name)
    target = make_computation_target(data)

    # Snapshot the initial target
    initial_target = {
        "col":        index_to_letter(target["col"]),
        "rows":       "-".join([index_to_letter(r) for r in target["rows"]]),
        "values":     "-".join([str(v) for v in target["values"]]),
        "correct_sum": target["correct_sum"],
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    csv_name  = f"{participant_id}_{block_name}_computation_trial{trial_num}_{timestamp}.csv"
    csv_path  = os.path.join(STIM_DIR, csv_name)
    save_dataset_csv(csv_path, data)

    static_stims, clickable_cells = build_table_stimuli(
        win, data, task_type="computation", difficulty=block_name
    )

    prompt_stim = visual.TextStim(
        win, text="", pos=(0, 0.84), color="black", height=0.038, wrapWidth=1.8
    )
    timer_stim = visual.TextStim(
        win, text="", pos=(0.78, 0.92), color="black", height=0.04
    )
    answer_label = visual.TextStim(
        win, text="Answer:", pos=(-0.40, -0.90), color="red",
        height=0.045, anchorHoriz="center"
    )
    answer_box = visual.Rect(
        win, width=0.42, height=0.08, pos=(0.00, -0.90),
        lineColor="black", fillColor="grey", lineWidth=2
    )
    answer_text_stim = visual.TextStim(
        win, text="", pos=(0.00, -0.90), color="black", height=0.045
    )
    feedback_stim = visual.TextStim(
        win, text="", pos=(0, -0.90), color="red", height=0.04
    )

    event.clearEvents()
    answer_text = ""
    feedback_text = ""
    feedback_until = None
    correct_count = 0
    response_count = 0 # correct submissions
    last_trigger_response = ""

    total_submissions = 0  # all ENTER presses with non-empty input
    incorrect_submissions = 0
    correct_rts = []  # RT of each correct submission
    question_num = 1   # 1-indexed, advances after each correct answer

    base_onset = TRIGGER_CODES[("computation", block_name, "onset")]
    base_response = TRIGGER_CODES[("computation", block_name, "response")]
    base_timeout = TRIGGER_CODES[("computation", block_name, "timeout")]

    onset_code = _make_trigger_code(base_onset, round_num)
    timeout_code = _make_trigger_code(base_timeout, round_num)

    def _prompt_text():
        return (
            f"Type the sum of Column {index_to_letter(target['col'])} "
            f"for Rows {index_to_letter(target['rows'][0])}, "
            f"{index_to_letter(target['rows'][1])}, and "
            f"{index_to_letter(target['rows'][2])}.\n"
            f"Press RETURN to submit. Backspace deletes.\n"
        )

    prompt_stim.text = _prompt_text()
    prompt_stim.draw()
    timer_stim.draw()
    draw_table_stimuli(static_stims, clickable_cells)
    answer_label.draw()
    answer_box.draw()
    answer_text_stim.draw()
    win.flip()

    trial_clock = core.Clock()
    send_trigger(base_onset, round_num=round_num,
                 label=f"computation_{block_name}_onset_r{round_num}")

    while trial_clock.getTime() < TRIAL_DURATION_SEC:
        time_left = max(0, TRIAL_DURATION_SEC - trial_clock.getTime())
        timer_stim.text = f"Time left: {format_time(time_left)}"
        timer_stim.color = "red" if time_left < 5 else "black"

        prompt_stim.text = _prompt_text()

        answer_box.lineColor = "black"
        answer_box.fillColor = None

        if feedback_until is not None and core.getTime() < feedback_until:
            feedback_stim.text = feedback_text
            if feedback_text == "Correct!":
                answer_box.lineColor = "green"
                answer_box.fillColor = "#003300"
            else:
                answer_box.lineColor = "red"
                answer_box.fillColor = "#330000"
        elif feedback_until is not None and core.getTime() >= feedback_until:
            feedback_text = ""
            feedback_stim.text = ""
            feedback_until = None

        prompt_stim.draw()
        timer_stim.draw()
        draw_table_stimuli(static_stims, clickable_cells)
        answer_label.draw()
        answer_box.draw()
        answer_text_stim.text = answer_text
        answer_text_stim.draw()
        if feedback_text:
            feedback_stim.draw()
        win.flip()

        keys = event.getKeys(keyList=[
            "escape", "return", "num_enter", "backspace",
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"
        ])

        for key in keys:
            if key == "escape":
                core.quit()
            elif key in "0123456789":
                answer_text += key
            elif key == "backspace":
                answer_text = answer_text[:-1]
            elif key in ["return", "num_enter"] and answer_text != "":
                submit_t   = trial_clock.getTime()
                submitted  = int(answer_text)
                is_correct = submitted == target["correct_sum"]

                total_submissions += 1
                if not is_correct:
                    incorrect_submissions += 1

                # per-response row 
                response_writer.writerow({
                    "participant_id": participant_id,
                    "trial_num": trial_num,
                    "round_num": round_num,
                    "trial_in_round": trial_in_round,
                    "task_type": "computation",
                    "difficulty": block_name,
                    "response_type": "submission",
                    "target_num": question_num,
                    "target_row": "-".join([index_to_letter(r) for r in target["rows"]]),
                    "target_col": index_to_letter(target["col"]),
                    "target_value": target["correct_sum"],
                    "target_color": "",
                    "response_row": "",
                    "response_col": "",
                    "response_value": "",
                    "response_color": "",
                    "correct": int(is_correct),
                    "rt_s": round(submit_t, 4),
                    "submitted_answer": submitted,
                    "correct_answer": target["correct_sum"],
                    "trigger_code": "",
                })

                if is_correct:
                    correct_count  += 1
                    response_count += 1
                    correct_rts.append(round(submit_t, 4))

                    trig_code = _make_trigger_code(base_response, round_num, response_count)
                    last_trigger_response = trig_code
                    send_trigger(base_response,
                                 round_num=round_num,
                                 response_count=response_count,
                                 label=f"computation_{block_name}_response_r{round_num}_n{response_count}")
                    feedback_text = "Correct!"
                    feedback_stim.color = "green"
                    feedback_until = core.getTime() + 1.0
                    answer_text = ""
                    target = make_computation_target(data)
                    question_num += 1
                else:
                    feedback_text = "Incorrect. Try again."
                    feedback_stim.color = "red"
                    feedback_until = core.getTime() + 1.0
                    answer_text = ""

        core.wait(0.01)

    trial_duration = trial_clock.getTime()
    send_trigger(base_timeout, round_num=round_num,
                 label=f"computation_{block_name}_timeout_r{round_num}")

    # ── Summary stats ─────────────────────────────────────────────────────
    accuracy = (correct_count / total_submissions) if total_submissions > 0 else ""
    mean_rt = round(statistics.mean(correct_rts), 4) if correct_rts else ""
    median_rt = round(statistics.median(correct_rts), 4) if correct_rts else ""
    min_rt = round(min(correct_rts), 4) if correct_rts else ""
    max_rt = round(max(correct_rts), 4) if correct_rts else ""
    iris = [round(correct_rts[i] - correct_rts[i-1], 4)
            for i in range(1, len(correct_rts))] if len(correct_rts) > 1 else []
    mean_iri = round(statistics.mean(iris), 4) if iris else ""

    return {
        # identity
        "participant_id": participant_id,
        "trial_num": trial_num,
        "round_num": round_num,
        "trial_in_round": trial_in_round,
        "task_type": "computation",
        "difficulty": block_name,
        # stimulus
        "stimulus_csv": csv_path,
        # first (initial) target
        "first_target_row": "",
        "first_target_col": initial_target["col"],
        "first_target_value": "",
        "first_target_color": "",
        # performance summary 
        "correct_count": correct_count,
        "total_clicks": "",
        "error_count": "",
        "accuracy": round(accuracy, 4) if accuracy != "" else "",
        "targets_presented": question_num,
        # (correct responses only, seconds)
        "rt_first_correct_s": correct_rts[0] if correct_rts else "",
        "rt_mean_s": mean_rt,
        "rt_median_s": median_rt,
        "rt_min_s": min_rt,
        "rt_max_s": max_rt,
        "mean_iri_s": mean_iri,
        # trial timing 
        "trial_duration_s": round(trial_duration, 4),
        # trigger codes 
        "trigger_onset": onset_code,
        "trigger_last_response":last_trigger_response,
        "trigger_timeout": timeout_code,
        # compact JSON (empty for computation)
        "target_sequence_json": "",
        "correct_rts_json": json.dumps(correct_rts),
        # computation-specific fields 
        "first_target_rows": initial_target["rows"],
        "first_target_values": initial_target["values"],
        "correct_sum": initial_target["correct_sum"],
        "total_submissions": total_submissions,
        "incorrect_submissions":incorrect_submissions,
    }


# =========================
# BLOCK LOGIC
# =========================

# Trial-level CSV fieldnames
TRIAL_FIELDNAMES = [
    # identity
    "participant_id", "trial_num", "round_num", "trial_in_round",
    "task_type", "difficulty",
    # stimulus
    "stimulus_csv",
    # first target (click/spatial: row+col+value+color; computation: col+rows+values+sum)
    "first_target_row", "first_target_col", "first_target_value", "first_target_color",
    "first_target_rows", "first_target_values", "correct_sum",
    # performance summary
    "correct_count", "total_clicks", "error_count", "accuracy",
    "total_submissions", "incorrect_submissions",
    "targets_presented",
    # RT summary (correct responses, seconds)
    "rt_first_correct_s", "rt_mean_s", "rt_median_s",
    "rt_min_s", "rt_max_s", "mean_iri_s",
    # trial timing
    "trial_duration_s",
    # triggers
    "trigger_onset", "trigger_last_response", "trigger_timeout",
    # compact JSON archives
    "target_sequence_json", "correct_rts_json",
]

# Response-level CSV fieldnames
RESPONSE_FIELDNAMES = [
    # identity
    "participant_id", "trial_num", "round_num", "trial_in_round",
    "task_type", "difficulty",
    # which target/question was active
    "response_type", # "click" | "submission"
    "target_num", # 1-indexed within trial
    # target info at time of response
    "target_row", "target_col", "target_value", "target_color",
    # what was actually clicked / submitted
    "response_row", "response_col", "response_value", "response_color",
    # outcome
    "correct", # 1 / 0
    "rt_s", # seconds since trial onset
    # computation only
    "submitted_answer", "correct_answer",
    # trigger (filled for correct responses only, where applicable)
    "trigger_code",
]


def run_randomized_trials(win, mouse, participant_id,
                          trial_writer, response_writer, results_fh, responses_fh):
    trial_sequence  = build_trial_sequence()
    trials_per_round = len(get_all_trial_conditions())

    for trial_num, (task_type, difficulty) in enumerate(trial_sequence, start=1):
        round_num      = ((trial_num - 1) // trials_per_round) + 1
        trial_in_round = ((trial_num - 1) %  trials_per_round) + 1

        is_first_trial_in_round = (trial_in_round == 1)
        is_last_trial_in_round  = (trial_in_round == trials_per_round)

        if is_first_trial_in_round:
            if EYE_TRACKING and el is not None:
                send_block_start(round_num)
                pylink.pumpDelay(50)

        show_fixation(win)

        send_trial_start(
            trial_num=trial_num,
            round_num=round_num,
            trial_in_round=trial_in_round,
            task_type=task_type,
            difficulty=difficulty
        )

        if EYE_TRACKING and el is not None:
            pylink.pumpDelay(20)

        if task_type in ["spatial", "visibility"]:
            result = run_click_trial(
                win=win, mouse=mouse,
                trial_num=trial_num, round_num=round_num, trial_in_round=trial_in_round,
                task_type=task_type, block_name=difficulty,
                participant_id=participant_id, response_writer=response_writer,
            )
        elif task_type == "computation":
            result = run_computation_trial(
                win=win,
                trial_num=trial_num, round_num=round_num, trial_in_round=trial_in_round,
                block_name=difficulty,
                participant_id=participant_id, response_writer=response_writer,
            )
        else:
            raise ValueError(f"Unknown task type: {task_type}")

        send_trial_end()

        if EYE_TRACKING and el is not None:
            pylink.pumpDelay(50)

        trial_writer.writerow(result)
        results_fh.flush()
        responses_fh.flush()

        if is_last_trial_in_round:
            send_block_end(round_num)
            if EYE_TRACKING and el is not None:
                pylink.pumpDelay(50)

        if is_last_trial_in_round and round_num < N_TRIAL_REPEATS:
            break_screen(win, block_name="Round complete", seconds=BREAK_DURATION_SEC)


# =========================
# MAIN
# =========================

def main():
    global el

    exp_info = {"participant_id": ""}
    dlg = gui.DlgFromDict(exp_info, title="Spreadsheet Task")
    if not dlg.OK:
        core.quit()

    participant_id = exp_info["participant_id"].strip()
    if participant_id == "":
        participant_id = "anon"

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_file   = os.path.join(RESULTS_DIR, f"{participant_id}_trials_{timestamp_str}.csv")
    responses_file = os.path.join(RESULTS_DIR, f"{participant_id}_responses_{timestamp_str}.csv")

    win = visual.Window(
        size=WINDOW_SIZE, fullscr=FULLSCREEN,
        color="white", units="norm", screen=0
    )

    if EYE_TRACKING:
        el = pylink.EyeLink("100.1.1.1")
        el.openDataFile(f"{participant_id}TT.edf")

        show_message(
            win,
            text="Eye tracking setup is about to begin.",
            height=0.05
        )

        genv = EyeLinkCoreGraphicsPsychoPy(el, win)
        pylink.openGraphicsEx(genv)

        el.sendCommand("sample_rate 1000")
        el.sendCommand("recording_parse_type = GAZE")
        el.sendCommand("calibration_type = HV9")

        el.doTrackerSetup()

        win.close()
        win = visual.Window(
            size=WINDOW_SIZE, fullscr=FULLSCREEN,
            color="white", units="norm", screen=0
        )

        show_message(
            win,
            text="Calibration complete.\n\nPress SPACE to continue.",
            wait_for_space=True,
            height=0.05
        )

    if USE_EEG:
        eeg_inlet = connect_eeg_stream()

        if eeg_inlet is None:
            no_eeg_msg = visual.TextStim(
                win,
                text=(
                    "WARNING: gNautilus EEG stream not detected.\n\n"
                    "Check that g.NEEDaccess is running and streaming.\n\n"
                    "Press Space to continue WITHOUT EEG, or ESC to quit."
                ),
                color="black", height=0.048, wrapWidth=1.6
            )
            no_eeg_msg.draw()
            win.flip()
            keys = event.waitKeys(keyList=["space", "escape"])
            if "escape" in keys:
                core.quit()
        else:
            if not check_eeg_stream_alive(eeg_inlet):
                print("[EEG] WARNING: Stream connected but samples are not flowing yet.")

    mouse = event.Mouse(win=win)
    mouse.setVisible(True)

    welcome_text = (
        "Welcome to the experiment!\n\n"
        "You will be performing multiple types of tasks that require\n"
        "you to read and interact with Excel-style tables.\n\n"
        "The following instructions section will prepare you for each task\n"
        "type before real trials begin.\n\n"
        "Please press SPACE to continue."
    )
    show_message(win, welcome_text, wait_for_space=True, height=0.052)

    if RUN_INSTRUCTION_PHASE:
        run_instruction_and_demo_phase(win, mouse)

    real_trials_intro = (
        f"You will now complete {N_TRIAL_REPEATS} shuffled rounds of trials.\n"
        f"Each trial will last up to {TRIAL_DURATION_SEC} seconds.\n"
        f"After each block, you will be able to take a short break.\n\n"
        f"Press SPACE to begin."
    )
    show_message(win, real_trials_intro, wait_for_space=True, height=0.05)

    if EYE_TRACKING and el is not None:
        el.setOfflineMode()
        pylink.pumpDelay(50)
        el.startRecording(1, 1, 1, 1)
        pylink.pumpDelay(500)

    with open(results_file, "w", newline="") as results_fh, \
         open(responses_file, "w", newline="") as responses_fh:

        trial_writer = csv.DictWriter(
            results_fh, fieldnames=TRIAL_FIELDNAMES,
            extrasaction="ignore", restval=""
        )
        trial_writer.writeheader()
        results_fh.flush()

        response_writer = csv.DictWriter(
            responses_fh, fieldnames=RESPONSE_FIELDNAMES,
            extrasaction="ignore", restval=""
        )
        response_writer.writeheader()
        responses_fh.flush()

        run_randomized_trials(
            win, mouse, participant_id,
            trial_writer, response_writer,
            results_fh, responses_fh
        )

    end_text = (
        "You have completed the study.\n\n"
        "Thank you so much for participating!\n\n"
        "Press SPACE to close."
    )
    show_message(win, end_text, wait_for_space=True, height=0.05)

    if EYE_TRACKING and el is not None:
        try:
            el.stopRecording()
        except Exception:
            pass

        pylink.pumpDelay(100)

        try:
            el.setOfflineMode()
            pylink.pumpDelay(50)
            el.closeDataFile()
            el.receiveDataFile(f"{participant_id}TT.edf", f"{participant_id}TT.edf")
        finally:
            try:
                pylink.closeGraphics()
            except Exception:
                pass
            el.close()

    if USE_EEG and eeg_inlet is not None:
        eeg_inlet.close_stream()
        print("[EEG] Stream closed.")

    win.close()
    core.quit()


if __name__ == "__main__":
    main()