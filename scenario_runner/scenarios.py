"""Scenario definitions.

Edit this file to add/remove/change scenarios, then run from the project root:

    python -m scenario_runner.run

Each scenario is a tuple:
    (label, helper_name, helper_kwargs)

Tuples whose label starts with "##" are treated as section headers by the
runner and printed as dividers -- no hand is evaluated.

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
    # =================================================================
    # 1. BTN OPEN (unopened pot, hero on the button)
    # =================================================================
    ("## BTN OPEN -- deep stacks",      None, None),

    # Premium
    ("AA BTN 100bb",          "btn_open",   dict(hero_cards="Ah Ad", stack=100)),
    ("KK BTN 100bb",          "btn_open",   dict(hero_cards="Kh Kd", stack=100)),
    ("AKs BTN 100bb",         "btn_open",   dict(hero_cards="As Ks", stack=100)),
    ("AKo BTN 100bb",         "btn_open",   dict(hero_cards="As Kd", stack=100)),

    # Broadway
    ("QJs BTN 100bb",         "btn_open",   dict(hero_cards="Qs Js", stack=100)),
    ("JTs BTN 100bb",         "btn_open",   dict(hero_cards="Jh Th", stack=100)),
    ("KQo BTN 100bb",         "btn_open",   dict(hero_cards="Kd Qc", stack=100)),
    ("ATo BTN 100bb",         "btn_open",   dict(hero_cards="Ad Tc", stack=100)),
    ("QTo BTN 100bb",         "btn_open",   dict(hero_cards="Qd Tc", stack=100)),

    # Suited connectors / gappers
    ("87s BTN 100bb",         "btn_open",   dict(hero_cards="8s 7s", stack=100)),
    ("76s BTN 100bb",         "btn_open",   dict(hero_cards="7s 6s", stack=100)),
    ("54s BTN 100bb",         "btn_open",   dict(hero_cards="5s 4s", stack=100)),
    ("T8o BTN 100bb",         "btn_open",   dict(hero_cards="Td 8c", stack=100)),
    ("97s BTN 100bb",         "btn_open",   dict(hero_cards="9d 7d", stack=100)),

    # Suited aces
    ("A5s BTN 100bb",         "btn_open",   dict(hero_cards="As 5s", stack=100)),
    ("A2s BTN 100bb",         "btn_open",   dict(hero_cards="Ah 2h", stack=100)),
    ("A9o BTN 100bb",         "btn_open",   dict(hero_cards="As 9d", stack=100)),

    # Suited kings / queens
    ("K9s BTN 100bb",         "btn_open",   dict(hero_cards="Ks 9s", stack=100)),
    ("K5s BTN 100bb",         "btn_open",   dict(hero_cards="Kh 5h", stack=100)),
    ("Q8s BTN 100bb",         "btn_open",   dict(hero_cards="Qs 8s", stack=100)),

    # Pairs
    ("TT BTN 100bb",          "btn_open",   dict(hero_cards="Th Td", stack=100)),
    ("55 BTN 100bb",          "btn_open",   dict(hero_cards="5h 5d", stack=100)),
    ("22 BTN 100bb",          "btn_open",   dict(hero_cards="2h 2d", stack=100)),

    # Clear folds
    ("72o BTN 100bb",         "btn_open",   dict(hero_cards="7d 2c", stack=100)),
    ("83o BTN 100bb",         "btn_open",   dict(hero_cards="8d 3c", stack=100)),
    ("J4o BTN 100bb",         "btn_open",   dict(hero_cards="Jd 4c", stack=100)),
    ("Q3o BTN 100bb",         "btn_open",   dict(hero_cards="Qh 3c", stack=100)),
    ("T2o BTN 100bb",         "btn_open",   dict(hero_cards="Td 2c", stack=100)),

    # =================================================================
    # 2. BTN OPEN -- short stacks (jam-or-fold)
    # =================================================================
    ("## BTN OPEN -- short stacks",     None, None),

    ("AA BTN 4bb",            "btn_open",   dict(hero_cards="Ah Ad", stack=4)),
    ("AA BTN 8bb",            "btn_open",   dict(hero_cards="Ah Ad", stack=8)),
    ("A5s BTN 8bb",           "btn_open",   dict(hero_cards="As 5s", stack=8)),
    ("A5s BTN 18bb",          "btn_open",   dict(hero_cards="As 5s", stack=18)),
    ("A5s BTN 35bb",          "btn_open",   dict(hero_cards="As 5s", stack=35)),
    ("KTo BTN 8bb",           "btn_open",   dict(hero_cards="Kd Tc", stack=8)),
    ("55 BTN 8bb",            "btn_open",   dict(hero_cards="5h 5d", stack=8)),
    ("72o BTN 8bb",           "btn_open",   dict(hero_cards="7d 2c", stack=8)),
    ("T9s BTN 8bb",           "btn_open",   dict(hero_cards="Td 9d", stack=8)),
    ("22 BTN 4bb",            "btn_open",   dict(hero_cards="2h 2d", stack=4)),

    # =================================================================
    # 3. BB VS LIMP (BTN completes, BB decides)
    # =================================================================
    ("## BB VS LIMP",                   None, None),

    ("AA BB vs limp 100bb",   "bb_vs_limp", dict(hero_cards="Ah Ad", stack=100)),
    ("AKo BB vs limp 100bb",  "bb_vs_limp", dict(hero_cards="As Kd", stack=100)),
    ("QTs BB vs limp 100bb",  "bb_vs_limp", dict(hero_cards="Qs Ts", stack=100)),
    ("87s BB vs limp 100bb",  "bb_vs_limp", dict(hero_cards="8s 7s", stack=100)),
    ("JTo BB vs limp 100bb",  "bb_vs_limp", dict(hero_cards="Jd Tc", stack=100)),
    ("K5s BB vs limp 100bb",  "bb_vs_limp", dict(hero_cards="Kh 5h", stack=100)),
    ("T7o BB vs limp 100bb",  "bb_vs_limp", dict(hero_cards="Td 7c", stack=100)),
    ("72o BB vs limp 100bb",  "bb_vs_limp", dict(hero_cards="7d 2c", stack=100)),
    ("53o BB vs limp 100bb",  "bb_vs_limp", dict(hero_cards="5d 3c", stack=100)),

    # =================================================================
    # 4. BB VS OPEN -- standard sizing (2.5x)
    # =================================================================
    ("## BB VS OPEN -- 2.5x standard",  None, None),

    # 3bet candidates
    ("AA BB vs 2.5x 100bb",   "bb_vs_open", dict(hero_cards="Ah Ad", open=2.5, stack=100)),
    ("AKs BB vs 2.5x 100bb",  "bb_vs_open", dict(hero_cards="As Ks", open=2.5, stack=100)),
    ("AQo BB vs 2.5x 100bb",  "bb_vs_open", dict(hero_cards="Ad Qc", open=2.5, stack=100)),
    ("A5s BB vs 2.5x 100bb",  "bb_vs_open", dict(hero_cards="As 5s", open=2.5, stack=100)),
    ("A3s BB vs 2.5x 100bb",  "bb_vs_open", dict(hero_cards="Ah 3h", open=2.5, stack=100)),
    ("KQs BB vs 2.5x 100bb",  "bb_vs_open", dict(hero_cards="Ks Qs", open=2.5, stack=100)),

    # Call candidates
    ("KQo BB vs 2.5x 100bb",  "bb_vs_open", dict(hero_cards="Kh Qc", open=2.5, stack=100)),
    ("KJo BB vs 2.5x 100bb",  "bb_vs_open", dict(hero_cards="Kd Jc", open=2.5, stack=100)),
    ("T9s BB vs 2.5x 100bb",  "bb_vs_open", dict(hero_cards="Td 9d", open=2.5, stack=100)),
    ("76s BB vs 2.5x 100bb",  "bb_vs_open", dict(hero_cards="7s 6s", open=2.5, stack=100)),
    ("54s BB vs 2.5x 100bb",  "bb_vs_open", dict(hero_cards="5h 4h", open=2.5, stack=100)),
    ("99 BB vs 2.5x 100bb",   "bb_vs_open", dict(hero_cards="9h 9d", open=2.5, stack=100)),
    ("44 BB vs 2.5x 100bb",   "bb_vs_open", dict(hero_cards="4h 4d", open=2.5, stack=100)),
    ("Q9o BB vs 2.5x 100bb",  "bb_vs_open", dict(hero_cards="Qd 9c", open=2.5, stack=100)),
    ("J8s BB vs 2.5x 100bb",  "bb_vs_open", dict(hero_cards="Jh 8h", open=2.5, stack=100)),

    # Fold candidates
    ("72o BB vs 2.5x 100bb",  "bb_vs_open", dict(hero_cards="7d 2c", open=2.5, stack=100)),
    ("83o BB vs 2.5x 100bb",  "bb_vs_open", dict(hero_cards="8d 3c", open=2.5, stack=100)),
    ("J3o BB vs 2.5x 100bb",  "bb_vs_open", dict(hero_cards="Jd 3c", open=2.5, stack=100)),
    ("T4o BB vs 2.5x 100bb",  "bb_vs_open", dict(hero_cards="Td 4c", open=2.5, stack=100)),

    # =================================================================
    # 5. BB VS OPEN -- MDF price sensitivity (same hand, varied sizes)
    # =================================================================
    ("## BB VS OPEN -- price sensitivity",  None, None),

    # 76s across sizes: should call standard, fold large
    ("76s BB vs 2x 100bb",    "bb_vs_open", dict(hero_cards="7s 6s", open=2.0, stack=100)),
    ("76s BB vs 3x 100bb",    "bb_vs_open", dict(hero_cards="7s 6s", open=3.0, stack=100)),
    ("76s BB vs 4x 100bb",    "bb_vs_open", dict(hero_cards="7s 6s", open=4.0, stack=100)),
    ("76s BB vs 5x 100bb",    "bb_vs_open", dict(hero_cards="7s 6s", open=5.0, stack=100)),

    # KQo across sizes
    ("KQo BB vs 2x 100bb",   "bb_vs_open", dict(hero_cards="Kh Qc", open=2.0, stack=100)),
    ("KQo BB vs 3x 100bb",   "bb_vs_open", dict(hero_cards="Kh Qc", open=3.0, stack=100)),
    ("KQo BB vs 4x 100bb",   "bb_vs_open", dict(hero_cards="Kh Qc", open=4.0, stack=100)),
    ("KQo BB vs 5x 100bb",   "bb_vs_open", dict(hero_cards="Kh Qc", open=5.0, stack=100)),

    # 83o vs small open (MDF widening test)
    ("83o BB vs 2x 100bb",   "bb_vs_open", dict(hero_cards="8d 3c", open=2.0, stack=100)),
    ("83o BB vs 4x 100bb",   "bb_vs_open", dict(hero_cards="8d 3c", open=4.0, stack=100)),

    # Premium immunity: AA should always raise regardless of size
    ("AA BB vs 2x 100bb",    "bb_vs_open", dict(hero_cards="Ah Ad", open=2.0, stack=100)),
    ("AA BB vs 5x 100bb",    "bb_vs_open", dict(hero_cards="Ah Ad", open=5.0, stack=100)),

    # =================================================================
    # 6. BB VS OPEN -- stack depth sensitivity
    # =================================================================
    ("## BB VS OPEN -- stack depth",    None, None),

    ("KQo BB vs 2.5x 12bb",  "bb_vs_open", dict(hero_cards="Kh Qc", open=2.5, stack=12)),
    ("KQo BB vs 2.5x 25bb",  "bb_vs_open", dict(hero_cards="Kh Qc", open=2.5, stack=25)),
    ("KQo BB vs 2.5x 50bb",  "bb_vs_open", dict(hero_cards="Kh Qc", open=2.5, stack=50)),
    ("T9s BB vs 2.5x 15bb",  "bb_vs_open", dict(hero_cards="Td 9d", open=2.5, stack=15)),
    ("T9s BB vs 2.5x 40bb",  "bb_vs_open", dict(hero_cards="Td 9d", open=2.5, stack=40)),

    # =================================================================
    # 7. BTN VS 3BET (hero opened, villain 3bets)
    # =================================================================
    ("## BTN VS 3BET",                  None, None),

    # 4bet value
    ("AA BTN vs 3bet 100bb",   "btn_vs_3bet", dict(hero_cards="Ah Ad", open=2.5, threeb=8.0, stack=100)),
    ("KK BTN vs 3bet 100bb",   "btn_vs_3bet", dict(hero_cards="Kh Kd", open=2.5, threeb=8.0, stack=100)),
    ("QQ BTN vs 3bet 100bb",   "btn_vs_3bet", dict(hero_cards="Qh Qd", open=2.5, threeb=8.0, stack=100)),
    ("AKo BTN vs 3bet 100bb",  "btn_vs_3bet", dict(hero_cards="As Kd", open=2.5, threeb=8.0, stack=100)),
    ("AKs BTN vs 3bet 100bb",  "btn_vs_3bet", dict(hero_cards="As Ks", open=2.5, threeb=8.0, stack=100)),
    ("A5s BTN vs 3bet 100bb",  "btn_vs_3bet", dict(hero_cards="As 5s", open=2.5, threeb=8.0, stack=100)),

    # Flat call
    ("JJ BTN vs 3bet 100bb",   "btn_vs_3bet", dict(hero_cards="Jh Jd", open=2.5, threeb=8.0, stack=100)),
    ("TT BTN vs 3bet 100bb",   "btn_vs_3bet", dict(hero_cards="Th Td", open=2.5, threeb=8.0, stack=100)),
    ("AQs BTN vs 3bet 100bb",  "btn_vs_3bet", dict(hero_cards="As Qs", open=2.5, threeb=8.0, stack=100)),
    ("AQo BTN vs 3bet 100bb",  "btn_vs_3bet", dict(hero_cards="Ad Qc", open=2.5, threeb=8.0, stack=100)),
    ("76s BTN vs 3bet 100bb",  "btn_vs_3bet", dict(hero_cards="7s 6s", open=2.5, threeb=8.0, stack=100)),
    ("T9s BTN vs 3bet 100bb",  "btn_vs_3bet", dict(hero_cards="Td 9d", open=2.5, threeb=8.0, stack=100)),
    ("98s BTN vs 3bet 100bb",  "btn_vs_3bet", dict(hero_cards="9s 8s", open=2.5, threeb=8.0, stack=100)),

    # Clear folds
    ("A3s BTN vs 3bet 100bb",  "btn_vs_3bet", dict(hero_cards="As 3s", open=2.5, threeb=8.0, stack=100)),
    ("72o BTN vs 3bet 100bb",  "btn_vs_3bet", dict(hero_cards="7d 2c", open=2.5, threeb=8.0, stack=100)),
    ("K5o BTN vs 3bet 100bb",  "btn_vs_3bet", dict(hero_cards="Kd 5c", open=2.5, threeb=8.0, stack=100)),

    # Stack depth variation
    ("76s BTN vs 3bet 20bb",   "btn_vs_3bet", dict(hero_cards="7s 6s", open=2.5, threeb=8.0, stack=20)),
    ("76s BTN vs 3bet 40bb",   "btn_vs_3bet", dict(hero_cards="7s 6s", open=2.5, threeb=8.0, stack=40)),
    ("AA BTN vs 3bet 15bb",    "btn_vs_3bet", dict(hero_cards="Ah Ad", open=2.5, threeb=8.0, stack=15)),

    # =================================================================
    # 8. BTN VS ISO (hero limped, villain raised)
    # =================================================================
    ("## BTN VS ISO (limped, facing raise)",  None, None),

    # Re-raise
    ("AA BTN vs iso 3.5x",     "btn_vs_iso", dict(hero_cards="Ah Ad", raise_to=3.5, stack=100)),
    ("QQ BTN vs iso 3.5x",     "btn_vs_iso", dict(hero_cards="Qh Qd", raise_to=3.5, stack=100)),
    ("AKo BTN vs iso 3.5x",    "btn_vs_iso", dict(hero_cards="As Kd", raise_to=3.5, stack=100)),

    # Flat call
    ("88 BTN vs iso 3.5x",     "btn_vs_iso", dict(hero_cards="8h 8d", raise_to=3.5, stack=100)),
    ("KQs BTN vs iso 3.5x",    "btn_vs_iso", dict(hero_cards="Ks Qs", raise_to=3.5, stack=100)),
    ("JTs BTN vs iso 3.5x",    "btn_vs_iso", dict(hero_cards="Jd Td", raise_to=3.5, stack=100)),
    ("T9s BTN vs iso 3.5x",    "btn_vs_iso", dict(hero_cards="Td 9d", raise_to=3.5, stack=100)),
    ("76s BTN vs iso 3.5x",    "btn_vs_iso", dict(hero_cards="7s 6s", raise_to=3.5, stack=100)),

    # Fold
    ("A5s BTN vs iso 3.5x",    "btn_vs_iso", dict(hero_cards="As 5s", raise_to=3.5, stack=100)),
    ("72o BTN vs iso 4x",      "btn_vs_iso", dict(hero_cards="7d 2c", raise_to=4.0, stack=100)),
    ("Q4o BTN vs iso 3.5x",    "btn_vs_iso", dict(hero_cards="Qd 4c", raise_to=3.5, stack=100)),

    # Larger iso size
    ("88 BTN vs iso 5x",       "btn_vs_iso", dict(hero_cards="8h 8d", raise_to=5.0, stack=100)),
    ("KQs BTN vs iso 5x",      "btn_vs_iso", dict(hero_cards="Ks Qs", raise_to=5.0, stack=100)),

    # =================================================================
    # 9. BB VS 4BET (hero 3bet, villain 4bets)
    # =================================================================
    ("## BB VS 4BET",                   None, None),

    # 5bet jam
    ("AA BB vs 4bet 100bb",    "bb_vs_4bet", dict(hero_cards="Ah Ad", open=2.5, threeb=8.0, fourb=20.0, stack=100)),
    ("KK BB vs 4bet 100bb",    "bb_vs_4bet", dict(hero_cards="Kh Kd", open=2.5, threeb=8.0, fourb=20.0, stack=100)),

    # Flat call
    ("QQ BB vs 4bet 100bb",    "bb_vs_4bet", dict(hero_cards="Qh Qd", open=2.5, threeb=8.0, fourb=20.0, stack=100)),
    ("JJ BB vs 4bet 100bb",    "bb_vs_4bet", dict(hero_cards="Jh Jd", open=2.5, threeb=8.0, fourb=20.0, stack=100)),
    ("AKo BB vs 4bet 100bb",   "bb_vs_4bet", dict(hero_cards="As Kd", open=2.5, threeb=8.0, fourb=20.0, stack=100)),
    ("AKs BB vs 4bet 100bb",   "bb_vs_4bet", dict(hero_cards="As Ks", open=2.5, threeb=8.0, fourb=20.0, stack=100)),
    ("AQs BB vs 4bet 100bb",   "bb_vs_4bet", dict(hero_cards="As Qs", open=2.5, threeb=8.0, fourb=20.0, stack=100)),

    # Clear folds
    ("JTs BB vs 4bet 100bb",   "bb_vs_4bet", dict(hero_cards="Jh Th", open=2.5, threeb=8.0, fourb=20.0, stack=100)),
    ("TT BB vs 4bet 100bb",    "bb_vs_4bet", dict(hero_cards="Th Td", open=2.5, threeb=8.0, fourb=20.0, stack=100)),
    ("76s BB vs 4bet 100bb",   "bb_vs_4bet", dict(hero_cards="7s 6s", open=2.5, threeb=8.0, fourb=20.0, stack=100)),
    ("72o BB vs 4bet 100bb",   "bb_vs_4bet", dict(hero_cards="7d 2c", open=2.5, threeb=8.0, fourb=20.0, stack=100)),
    ("A5s BB vs 4bet 100bb",   "bb_vs_4bet", dict(hero_cards="As 5s", open=2.5, threeb=8.0, fourb=20.0, stack=100)),

    # Stack depth
    ("AA BB vs 4bet 30bb",     "bb_vs_4bet", dict(hero_cards="Ah Ad", open=2.5, threeb=8.0, fourb=20.0, stack=30)),
]
