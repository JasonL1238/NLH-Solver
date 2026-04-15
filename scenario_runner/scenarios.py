"""Scenario definitions.

Edit this file to add/remove/change scenarios, then run from the project root:

    python -m scenario_runner.run

Each scenario is a tuple:
    (label, helper_name, helper_kwargs)

Available helpers and their kwargs:
    "btn_open"       -> hero_cards, stack
    "bb_vs_limp"     -> hero_cards, stack
    "bb_vs_open"     -> hero_cards, open, stack
    "btn_vs_iso"     -> hero_cards, raise_to, stack
    "btn_vs_3bet"    -> hero_cards, open, threeb, stack
    "bb_vs_4bet"     -> hero_cards, open, threeb, fourb, stack

Card format: "Xs Yd" where X/Y = rank (2-9, T, J, Q, K, A) and s/d/h/c = suit.
"""

SCENARIOS = [
    # --- BTN unopened ---
    ("AA BTN 100bb",          "btn_open",   dict(hero_cards="Ah Ad", stack=100)),
    ("AKs BTN 100bb",         "btn_open",   dict(hero_cards="As Ks", stack=100)),
    ("AKo BTN 100bb",         "btn_open",   dict(hero_cards="As Kd", stack=100)),
    ("QJs BTN 100bb",         "btn_open",   dict(hero_cards="Qs Js", stack=100)),
    ("JTs BTN 100bb",         "btn_open",   dict(hero_cards="Jh Th", stack=100)),
    ("87s BTN 100bb",         "btn_open",   dict(hero_cards="8s 7s", stack=100)),
    ("A5s BTN 100bb",         "btn_open",   dict(hero_cards="As 5s", stack=100)),
    ("K9s BTN 100bb",         "btn_open",   dict(hero_cards="Ks 9s", stack=100)),
    ("T8o BTN 100bb",         "btn_open",   dict(hero_cards="Td 8c", stack=100)),
    ("72o BTN 100bb",         "btn_open",   dict(hero_cards="7d 2c", stack=100)),
    ("83o BTN 100bb",         "btn_open",   dict(hero_cards="8d 3c", stack=100)),
    ("J4o BTN 100bb",         "btn_open",   dict(hero_cards="Jd 4c", stack=100)),
    ("55 BTN 100bb",          "btn_open",   dict(hero_cards="5h 5d", stack=100)),
    ("22 BTN 100bb",          "btn_open",   dict(hero_cards="2h 2d", stack=100)),

    # --- BTN short-stack ---
    ("AA BTN 8bb",            "btn_open",   dict(hero_cards="Ah Ad", stack=8)),
    ("A5s BTN 8bb",           "btn_open",   dict(hero_cards="As 5s", stack=8)),
    ("A5s BTN 18bb",          "btn_open",   dict(hero_cards="As 5s", stack=18)),
    ("A5s BTN 35bb",          "btn_open",   dict(hero_cards="As 5s", stack=35)),
    ("72o BTN 8bb",           "btn_open",   dict(hero_cards="7d 2c", stack=8)),

    # --- BB vs limp ---
    ("87s BB vs limp 100bb",  "bb_vs_limp", dict(hero_cards="8s 7s", stack=100)),
    ("AKo BB vs limp 100bb",  "bb_vs_limp", dict(hero_cards="As Kd", stack=100)),
    ("72o BB vs limp 100bb",  "bb_vs_limp", dict(hero_cards="7d 2c", stack=100)),
    ("QTs BB vs limp 100bb",  "bb_vs_limp", dict(hero_cards="Qs Ts", stack=100)),

    # --- BB vs open raise ---
    ("AKs BB vs 2.5x 100bb",  "bb_vs_open", dict(hero_cards="As Ks", open=2.5, stack=100)),
    ("AQo BB vs 2.5x 100bb",  "bb_vs_open", dict(hero_cards="Ad Qc", open=2.5, stack=100)),
    ("KQo BB vs 2.5x 100bb",  "bb_vs_open", dict(hero_cards="Kh Qc", open=2.5, stack=100)),
    ("T9s BB vs 2.5x 100bb",  "bb_vs_open", dict(hero_cards="Td 9d", open=2.5, stack=100)),
    ("76s BB vs 2.5x 100bb",  "bb_vs_open", dict(hero_cards="7s 6s", open=2.5, stack=100)),
    ("72o BB vs 2.5x 100bb",  "bb_vs_open", dict(hero_cards="7d 2c", open=2.5, stack=100)),
    ("83o BB vs 4x 100bb",    "bb_vs_open", dict(hero_cards="8d 3c", open=4.0, stack=100)),
    ("KQo BB vs 2.5x 12bb",   "bb_vs_open", dict(hero_cards="Kh Qc", open=2.5, stack=12)),
    ("KQo BB vs 2.5x 25bb",   "bb_vs_open", dict(hero_cards="Kh Qc", open=2.5, stack=25)),
    ("KQo BB vs 2.5x 50bb",   "bb_vs_open", dict(hero_cards="Kh Qc", open=2.5, stack=50)),

    # --- BTN vs 3bet ---
    ("QQ BTN vs 3bet 100bb",   "btn_vs_3bet", dict(hero_cards="Qh Qd", open=2.5, threeb=8.0, stack=100)),
    ("AKo BTN vs 3bet 100bb",  "btn_vs_3bet", dict(hero_cards="As Kd", open=2.5, threeb=8.0, stack=100)),
    ("A3s BTN vs 3bet 100bb",  "btn_vs_3bet", dict(hero_cards="As 3s", open=2.5, threeb=8.0, stack=100)),
    ("76s BTN vs 3bet 100bb",  "btn_vs_3bet", dict(hero_cards="7s 6s", open=2.5, threeb=8.0, stack=100)),
    ("76s BTN vs 3bet 20bb",   "btn_vs_3bet", dict(hero_cards="7s 6s", open=2.5, threeb=8.0, stack=20)),
    ("76s BTN vs 3bet 40bb",   "btn_vs_3bet", dict(hero_cards="7s 6s", open=2.5, threeb=8.0, stack=40)),
    ("72o BTN vs 3bet 100bb",  "btn_vs_3bet", dict(hero_cards="7d 2c", open=2.5, threeb=8.0, stack=100)),

    # --- BTN vs iso after limp ---
    ("88 BTN vs iso 3.5x 100bb",  "btn_vs_iso", dict(hero_cards="8h 8d", raise_to=3.5, stack=100)),
    ("A5s BTN vs iso 3.5x 100bb", "btn_vs_iso", dict(hero_cards="As 5s", raise_to=3.5, stack=100)),
    ("72o BTN vs iso 4x 100bb",   "btn_vs_iso", dict(hero_cards="7d 2c", raise_to=4.0, stack=100)),

    # --- BB vs 4bet ---
    ("AKo BB vs 4bet 100bb",  "bb_vs_4bet", dict(hero_cards="As Kd", open=2.5, threeb=8.0, fourb=20.0, stack=100)),
    ("QQ BB vs 4bet 100bb",   "bb_vs_4bet", dict(hero_cards="Qh Qd", open=2.5, threeb=8.0, fourb=20.0, stack=100)),
    ("JTs BB vs 4bet 100bb",  "bb_vs_4bet", dict(hero_cards="Jh Th", open=2.5, threeb=8.0, fourb=20.0, stack=100)),
    ("72o BB vs 4bet 100bb",  "bb_vs_4bet", dict(hero_cards="7d 2c", open=2.5, threeb=8.0, fourb=20.0, stack=100)),
]
