def(elevated, or([papule, plaque, nodule, domeshaped])).
def(pigmented, or([brown, black])).
def(multicolor, atleast(3, [brown, black, white, purple, yellow, erythema])).

rule('R-MN-3', mn, n, b, or([ulcer, friable])).
rule('R-MN-4', mn, e, b, and([pigmented, multicolor])).
rule('R-MEL-4', mel, n, b, not(or([brown, black, erythema]))).
rule('R-MEL-5', mel, e, b, and([warty, comedo])).
rule('R-BCC-5', bcc, e, b, or([vesicle, comedo])).
rule('R-SCC-4', scc, n, b, not(or([scale, crust, ulcer, exophytic]))).
rule('R-SCC-5', scc, e, b, and([translucent, telangiectasia])).
rule('R-SK-3', sk, n, b, or([not(elevated), ulcer]))
rule('R-SK-4', sk, e, b, or([translucent, vesicle, pustule])).
rule('R-VV-3', vv, n, a, not(warty)).
rule('R-VV-4', vv, e, b, or([translucent, telangiectasia, comedo])).
rule('R-ACN-4', acn, n, a, not(or([comedo, papule, pustule, nodule]))).
rule('R-ACN-5', acn, e, b, or([vesicle, bulla, lichenification, warty])).
rule('R-FOL-3', fol, n, a, not(or([pustule, papule]))).
rule('R-FOL-4', fol, e, b, comedo).
rule('R-ECZ-4', ecz, n, b, not(or([erythema, lichenification, xerosis]))).
rule('R-ECZ-5', ecz, e, b, or([comedo, translucent, warty, atrophy])).
rule('R-ACD-3', acd, n, a, not(erythema)).
rule('R-ACD-4', acd, e, b, or([comedo, translucent, warty, atrophy])).
rule('R-PHD-4', phd, e, b, or([comedo, warty, translucent])).
rule('R-LP-5', lp, e, b, or([vesicle, bulla, pustule, comedo]))
rule('R-LE-4', le, e, b, or([comedo, warty, pustule]))
rule('R-SCL-4', scl, e, b, or([scale, vesicle, pustule, exudate])).
rule('R-GA-2', ga, n, b, or([scale, vesicle, pustule])).
rule('R-GA-3', ga, e, b, or([comedo, warty, ulcer])).
rule('R-SAR-4', sar, e, b, or([vesicle, pustule, exudate, excoriation])).
rule('R-PRP-3', prp, n, b, not(scale)).
rule('R-PRP-4', prp, e, b, or([vesicle, bulla, pustule, comedo, atrophy])).
rule('R-EM-3', em, n, b, or([not(erythema), not(or([papule, plaque]))])).
rule('R-EM-4', em, e, b, or([scale, lichenification, comedo, warty])).
rule('R-DE-4', de, e, b, or([comedo, warty, lichenification, translucent])).
rule('R-ND-3', nd, n, b, not(or([erythema, ulcer])))
rule('R-ND-4', nd, e, b, or([comedo, warty, lichenification, scale])).
rule('R-SCB-3', scb, n, b, not(or([papule, nodule, crust, excoriation]))).
rule('R-SCB-4', scb, e, b, or([comedo, warty, translucent, atrophy])).
rule('R-NI-4', ni, e, b, or([comedo, warty, translucent])).

srule('R-MEL-1', mel, a, and([pigmented, multicolor, or([macule, patch, plaque])])).
srule('R-BCC-1', bcc, a, and([translucent, telangiectasia, elevated])).
srule('R-SK-2', sk, a, and([comedo, pigmented, or([warty, plaque])])).
srule('R-ECZ-1', ecz, a, and([lichenification, excoriation, or([xerosis, scale])])).

drule('R-DD-1', mn, mel, a, or([multicolor, ulcer, friable]), mel).
drule('R-DD-2', bcc, scc, a, translucent, bcc).
drule('R-DD-3', sk, mel, b, and([multicolor, not(comedo)]), mel).
drule('R-DD-4', acn, fol, a, comedo, acn).

grule('R-G-1', completeness).
grule('R-G-2', mutual_exclusion).
