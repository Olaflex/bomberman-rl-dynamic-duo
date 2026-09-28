
from contextlib import contextmanager
from dataclasses import dataclass, field as dc_field
from typing import Dict, List

import settings as s


@dataclass
class Stage:

    name: str
    scenario: str
    opponents: List[str] = dc_field(default_factory=list)
    config: Dict[str, float] = None
    skill: str = ''
    gate: str = ''


STAGES: Dict[str, Stage] = {
    'S0': Stage('S0', 'coin-heaven', [], None,
                'move efficiently, no invalid actions',
                'steps-per-coin <= 1.3 x GBR and invalid/step < 0.5%'),
    'S1': Stage('S1', 'ladder-s1', [], {'CRATE_DENSITY': 0.35, 'COIN_COUNT': 9},
                'drop a bomb and survive it',
                'bombs/episode >= 1.0 and suicides/episode < g and '
                'forced-bomb survival >= 0.95 and S0 held'),
    'S2': Stage('S2', 'loot-crate', [], None,
                'bomb crates, collect what they reveal',
                'crates/episode >= 8 and S1 and S0 held'),
    'S3': Stage('S3', 'classic', ['peaceful_agent', 'coin_collector_agent'], None,
                'score against a moving target without dying to it',
                'kills/episode > 0.2 and deaths-by-other/episode < 0.5'),
    'S4': Stage('S4', 'classic', ['rule_based_agent'] * 3, None,
                'the real game',
                'score mu above rule_based_agent'),
}


@contextmanager
def stage_scenarios(stage: Stage):
    injected = stage.config is not None and stage.scenario not in s.SCENARIOS
    if injected:
        s.SCENARIOS[stage.scenario] = dict(stage.config)
    try:
        yield stage.scenario
    finally:
        if injected:
            s.SCENARIOS.pop(stage.scenario, None)


def seats(stage: Stage, agent: str) -> List[str]:
    return [agent, *stage.opponents]
