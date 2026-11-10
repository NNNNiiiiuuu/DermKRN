:- dynamic val/2, xrule/3, dkb/4.

tv(and(L), V) :- !, maplist(tv, L, Vs), min_list(Vs, V).
tv(or(L), V) :- !, maplist(tv, L, Vs), max_list(Vs, V).
tv(not(P), V) :- !, tv(P, X), V is 1 - X.
tv(atleast(N, L), V) :- !, maplist(tv, L, Vs), cnt(1, Vs, T), cnt(0.5, Vs, U),
    ( T >= N -> V = 1 ; T + U < N -> V = 0 ; V = 0.5 ).
tv(C, V) :- def(C, P), !, tv(P, V)
tv(C, V) :- val(C, V), !.
tv(_, 0.5).

cnt(X, L, N) :- aggregate_all(count, (member(Y, L), Y =:= X), N).

crule(R, K, P) :- rule(R, K, _, _, P).
crule(R, K, P) :- xrule(R, K, P).

out(K) :- crule(_, K, P), tv(P, 1), !.

infer(F, E, S, D, X) :-
    retractall(val(_, _)),
    forall(member(C-V, F), assertz(val(C, V)))
    findall([R, K], (crule(R, K, P), tv(P, 1)), E),
    findall(K, (srule(_, K, _, P), tv(P, 1), \+ out(K)), S0), sort(S0, S),
    findall([R, K], (drule(R, A, B, _, P, K), (memberchk(A, S) ; memberchk(B, S)), tv(P, 1), \+ out(K)), D),
    findall(I, (dkb(I, _, U, W), tv(U, 1), tv(W, 1)), X).

contra(K, U, W) :-
    retractall(val(_, _)), assertz(val(U, 1)), assertz(val(W, 1)),
    srule(_, K, _, P), tv(P, 1), !.

leaf(and(L), C) :- !, member(P, L), leaf(P, C).
leaf(or(L), C) :- !, member(P, L), leaf(P, C).
leaf(not(P), C) :- !, leaf(P, C)
leaf(atleast(_, L), C) :- !, member(P, L), leaf(P, C).
leaf(A, C) :- def(A, P), !, leaf(P, C).
leaf(A, A).

leaves(R, L) :- crule(R, _, P), setof(C, leaf(P, C), L).

negated(R) :- rule(R, _, _, _, not(_)).

dup(N) :- retractall(xrule(_, _, _)),
    forall((rule(R, K, _, _, P), between(2, N, I)), (format(atom(X), '~w#~w', [R, I]), assertz(xrule(X, K, P)))).
