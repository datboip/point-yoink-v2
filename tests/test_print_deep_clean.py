"""Print starts from the clean point model and still resets across every preset."""
import pytest
from scanner.stages import params as PR

def normalized(r):
    return dict(params=r.get('params',{}),ops={k:[{q:v for q,v in o.items() if q!='id'} for o in ops] for k,ops in r.get('ops',{}).items()})

def test_print_uses_the_deep_clean_input_and_finishes_it():
    clean=PR.apply_preset({},'revo-device');printed=PR.apply_preset({},'print')
    assert PR.method_of(printed,'fused')=='points'
    assert clean['params']==printed['params']
    assert normalized(clean)['ops']['fused.points']==normalized(printed)['ops']['fused.points']
    assert printed['ops']['meshed'][-1]['tool']=='finish'

@pytest.mark.parametrize('previous',list(PR.PRESETS))
@pytest.mark.parametrize('target',list(PR.PRESETS))
def test_all_36_preset_orders_reset(previous,target):
    assert normalized(PR.apply_preset(PR.apply_preset({},previous),target))==normalized(PR.apply_preset({},target))
