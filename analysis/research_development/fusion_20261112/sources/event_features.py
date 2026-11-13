"""Band, character, card and strictly prior-event context for forecasting."""
from __future__ import annotations

from collections import Counter
from datetime import datetime
import json
from pathlib import Path

import numpy as np

from analysis.backtest import ROOT, TZ, snapshot


class EventFeatures:
    def __init__(self, events, characters, bands, cards):
        self.events, self.characters, self.bands, self.cards = events, characters, bands, cards
        self.parties = {}
        for event, info in events.items():
            character_ids = sorted({int(c['characterId']) for c in info.get('characters', [])})
            band_counts = Counter(int(characters[str(c)]['bandId']) for c in character_ids
                                  if str(c) in characters and characters[str(c)].get('bandId') is not None)
            total = sum(band_counts.values())
            self.parties[int(event)] = {'characters': character_ids,
                'bands': {band: count / total for band, count in band_counts.items()} if total else {}}
        active = [self.parties[e] for e in self.parties if e <= 328]
        self.character_ids = sorted({c for party in active for c in party['characters']})
        self.band_ids = sorted({b for party in active for b in party['bands']})
        self._calendar_cache = {}

    @classmethod
    def load(cls, directory=ROOT/'data/metadata'):
        values = [json.loads((Path(directory)/f'{name}.json').read_text())
                  for name in ['events','characters','bands','cards']]
        return cls(*values)

    def band_only(self, curve):
        party = self.parties[curve.event]
        bands = party['bands']
        group = str(next(iter(bands))) if len(bands) == 1 else ('mixed' if bands else 'unknown')
        result = {'band_group': group, 'type_band': f'{curve.kind}|{group}',
                  'band_count': len(bands), 'character_count': len(party['characters']),
                  'mixed_band': float(len(bands) > 1)}
        result.update({f'band_share_{band}': bands.get(band, 0.) for band in self.band_ids})
        return result

    def static(self, curve, as_of):
        info = self.events[str(curve.event)]
        party = self.parties[curve.event]
        result = self.band_only(curve)
        result['attribute'] = '|'.join(sorted(a['attribute'] for a in info.get('attributes', []))) or 'unknown'
        result.update({f'participant_{c}': float(c in party['characters']) for c in self.character_ids})
        available = []
        for index, member in enumerate(info.get('members', [])):
            card = self.cards.get(str(member['situationId']))
            if card is None:
                continue
            releases = card.get('releasedAt', [])
            released = releases[3] if len(releases) > 3 else None
            if released is not None and str(released).isdigit() and int(released) <= as_of:
                available.append((member.get('seq', index + 1), card))
        max_rarity = max((card['rarity'] for _, card in available), default=0)
        featured = {int(card['characterId']) for _, card in available if card['rarity'] == max_rarity}
        first = sorted(available, key=lambda pair: (-pair[1]['rarity'], pair[0]))
        result.update({'available_member_cards': len(available),
            'five_star_member_cards': sum(card['rarity'] == 5 for _, card in available),
            'limited_member_cards': sum(card.get('type') == 'limited' for _, card in available),
            'fes_member_cards': sum(card.get('type') in ['dreamfes','kirafes'] for _, card in available),
            'featured_character': str(first[0][1]['characterId']) if first else 'unknown'})
        result.update({f'featured_character_{c}': float(c in featured) for c in self.character_ids})
        if curve.event not in self._calendar_cache:
            start = datetime.fromtimestamp(curve.start / 1000, TZ)
            end = datetime.fromtimestamp(curve.end / 1000, TZ)
            hours = np.arange(curve.start, curve.end, 3600000)
            weekends = np.array([datetime.fromtimestamp(t / 1000, TZ).weekday() >= 5 for t in hours])
            self._calendar_cache[curve.event] = (hours, weekends, {
                'start_month_sin': np.sin(2*np.pi*start.month/12),
                'start_month_cos': np.cos(2*np.pi*start.month/12),
                'start_weekday': start.weekday(), 'end_weekday': end.weekday(),
                'total_weekend_fraction': float(weekends.mean()) if len(weekends) else 0.,
                'long_event': float(curve.duration > 12*86400000)})
        hours, weekends, calendar = self._calendar_cache[curve.event]
        result.update(calendar)
        remaining = weekends[hours >= as_of]
        result['remaining_weekend_fraction'] = float(remaining.mean()) if len(remaining) else 0.
        return result

    def historical(self, sample, curves):
        """Every target, including training targets, gets causal aggregates."""
        target = sample['curve']
        party = self.parties[target.event]
        target_characters = set(party['characters'])
        attribute = {a['attribute'] for a in self.events[str(target.event)].get('attributes', [])}
        rows = []
        for past in curves:
            if past.event == target.event or past.rank != target.rank or past.end + 15*60000 >= target.start:
                continue
            if past.final <= 0:
                continue
            other = self.parties[past.event]
            other_characters = set(other['characters'])
            overlap = len(target_characters & other_characters) / max(1, len(target_characters | other_characters))
            band_overlap = sum(value*other['bands'].get(band,0.) for band,value in party['bands'].items())
            other_attribute = {a['attribute'] for a in self.events[str(past.event)].get('attributes', [])}
            days = (target.start-past.end)/86400000
            weight = np.exp(-days/180)
            heat = np.log1p(past.final/(past.duration/3600000))
            timestamp = past.start+sample['progress']*past.duration
            correction = np.nan
            if past.times[0] <= timestamp <= past.times[-1]:
                current = float(np.interp(timestamp,past.times,past.scores))
                if current > 0:
                    correction = np.log(past.final/current)+np.log(sample['progress'])
            rows.append({'weight':weight,'heat':heat,'correction':correction,'end':past.end,
                'type':float(past.kind==target.kind),'band':band_overlap,'character':overlap,
                'type_band':float(past.kind==target.kind)*band_overlap,
                'attribute':float(attribute==other_attribute)})
        result = {'history_rank_events':len(rows)}
        if not rows:
            result.update({'historical_heat_global':np.nan,'historical_correction_global':0.,
                           'recent_heat_shift':0.,'recent_correction_shift':0.})
            for group in ['type','band','character','type_band','attribute']:
                result.update({f'history_{group}_weight':0.,f'history_{group}_heat_delta':0.,
                               f'history_{group}_correction':0.})
            return result
        weights = np.array([r['weight'] for r in rows])
        heat = np.array([r['heat'] for r in rows])
        correction = np.array([r['correction'] for r in rows])
        valid = np.isfinite(correction)
        heat_prior = float(np.average(heat,weights=weights))
        correction_prior = float(np.average(correction[valid],weights=weights[valid])) if valid.any() else 0.
        result.update({'historical_heat_global':heat_prior,'historical_correction_global':correction_prior})
        for group in ['type','band','character','type_band','attribute']:
            group_weights = weights*np.array([r[group] for r in rows])
            total = group_weights.sum()
            result[f'history_{group}_weight'] = float(total)
            result[f'history_{group}_heat_delta'] = float((np.dot(group_weights,heat)+3*heat_prior)/(total+3)-heat_prior)
            result[f'history_{group}_correction'] = float((np.dot(group_weights[valid],correction[valid])+3*correction_prior)/(group_weights[valid].sum()+3))
        latest = np.argsort([r['end'] for r in rows])[-5:]
        result['recent_heat_shift'] = float(np.mean(heat[latest])-heat_prior)
        recent_valid = [i for i in latest if valid[i]]
        result['recent_correction_shift'] = float(np.mean(correction[recent_valid])-correction_prior) if recent_valid else 0.
        return result

    def peer(self, sample, curves):
        result = {'peer_available':0.,'log_peer_current_ratio':np.nan,
                  'peer_6h_relative':np.nan,'peer_24h_relative':np.nan,'peer_24h_rate_difference':np.nan}
        target = sample['curve']
        other = next((c for c in curves if c.event==target.event and c.rank!=target.rank),None)
        value = snapshot(other,sample['now']) if other is not None else None
        if value is None:
            return result
        result.update({'peer_available':1.,'log_peer_current_ratio':float(np.log(value['score']/sample['score'])),
            'peer_6h_relative':value['features']['rate_6h_relative'],
            'peer_24h_relative':value['features']['rate_24h_relative'],
            'peer_24h_rate_difference':value['features']['rate_24h_relative']-sample['features']['rate_24h_relative']})
        return result

    def describe(self,event):
        party=self.parties[event]
        names={str(band):self.bands[str(band)]['bandName'][3] or self.bands[str(band)]['bandName'][0]
               for band in party['bands']}
        return {'event_id':event,'characters':party['characters'],'band_shares':party['bands'],'band_names':names,
                'band_group':'single' if len(names)==1 else 'mixed'}
