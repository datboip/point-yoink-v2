# SPDX-License-Identifier: AGPL-3.0-or-later
"""Portable build preferences. Saved presets never include scans or manual edits."""
import json
import math
import os
from pathlib import Path
import tempfile

DEFAULTS = dict(method='tsdf', detail=0.4, cleanup=3.0, smooth=0, finish=False)


def validate(settings):
    if not isinstance(settings, dict): raise ValueError('Preset settings must be an object.')
    if set(settings) - set(DEFAULTS): raise ValueError('Unknown preset setting.')
    result = dict(DEFAULTS, **settings)
    if result['method'] not in ('tsdf','points','plus'): raise ValueError('Choose a build method.')
    for key, low, high in [('detail',0.1,2.0),('cleanup',0,100),('smooth',0,20)]:
        value = result[key]
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError('Invalid '+key+' value.')
    if int(result['smooth']) != result['smooth']: raise ValueError('Smooth passes must be a whole number.')
    result['smooth'] = int(result['smooth'])
    if not isinstance(result['finish'],bool): raise ValueError('Finish must be on or off.')
    return result


def recipe_fragment(settings):
    s = validate(settings)
    fused, mesh = [], []
    # loose pieces go where Classic removes them, on Fused, so the Fused view is already clean (in Meshed it showed floaters)
    if s['cleanup'] > 0: fused.append(dict(tool='isolation',params={'rate':s['cleanup']}))
    if s['smooth'] > 0: mesh.append(dict(tool='smooth',params={'strength':30.0,'times':s['smooth']}))
    if s['finish']: mesh.append(dict(tool='finish',params={}))
    # an explicit list (empty for cleanup 0) keeps the method's hidden default cleanup from overriding the choice
    return {'set': {'fused.method':s['method'], 'fused.point_distance':s['detail'],
                    'fused.frames_per_side':300,'meshed.method':'auto'},
            'reset_nodes':['fused','meshed'], 'ops':{'fused.'+s['method']:fused, 'meshed':mesh}}


def validate_fragment(fragment):
    """Only accept fragments that can be produced by the portable preferences."""
    try:
        params=fragment['set']
        ops=list(fragment['ops'].get('fused.'+params['fused.method'],[]))+list(fragment['ops']['meshed'])   # older saves kept cleanup in Meshed
        settings=dict(method=params['fused.method'],detail=params['fused.point_distance'],cleanup=0,smooth=0,finish=False)
        for op in ops:
            if op['tool']=='isolation': settings['cleanup']=op['params']['rate']
            elif op['tool']=='smooth': settings['smooth']=op['params']['times']
            elif op['tool']=='finish': settings['finish']=True
            else: raise ValueError('This edit cannot be saved in a preset.')
        clean=recipe_fragment(settings)
        if set(fragment) != set(clean) or set(fragment['ops']) - {'meshed','fused.'+params['fused.method']}: raise ValueError('Unsupported preset settings.')
        return clean
    except (KeyError,TypeError,IndexError) as exc: raise ValueError('Invalid preset.') from exc


def store_path():
    config=Path(os.environ.get('POINTYOINK_CONFIG') or Path.home()/'.config/pointyoink/config.json')
    return config.parent/'build-presets.json'


def load_presets():
    path=store_path()
    if not path.exists(): return {}
    data=json.loads(path.read_text())
    if not isinstance(data,dict): raise ValueError('Invalid saved preset file.')
    return {name:validate(settings) for name,settings in data.items()}


def save_preset(name,settings):
    name=str(name).strip()
    if not name or len(name)>60 or any(ord(c)<32 for c in name): raise ValueError('Use a name from 1 to 60 letters.')
    data=load_presets(); data[name]=validate(settings)
    path=store_path(); path.parent.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(prefix='.presets-',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as stream: json.dump(data,stream,indent=2)
        os.replace(temp,path)
    finally:
        if os.path.exists(temp): os.unlink(temp)
    return name
