# Horses puzzle: 2nd and 3rd fastest

**Setup:** 25 horses, at most 5 per race, no stopwatch (a race only gives finishing order).

**Answer: 7 races in total give 1st, 2nd and 3rd.** The 7th race decides 2nd and 3rd.

## Races 1–6 (already established: 1st place)
- **Races 1–5:** split the horses into groups A–E and race each group. Inside each group, label by finish: A1 > A2 > … > A5.
- **Race 6:** race the five group winners. Rename the groups by this result so that A1 > B1 > C1 > D1 > E1.
- **A1 is the fastest overall.**

## Who can still be 2nd or 3rd?
A horse can finish in position *k* only if fewer than *k* horses are known to be faster than it.

Number of horses known to be faster, after race 6:

|   | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| A | 0 | 1 | 2 | 3 | 4 |
| B | 1 | 2 | 3 | 4 | 5 |
| C | 2 | 3 | 4 | 5 | 6 |
| D | 3 | 4 | 5 | 6 | 7 |
| E | 4 | 5 | 6 | 7 | 8 |

(For horse X*i*, this is (i − 1) horses above it in its own group + the winners that beat X1 in race 6.)

- **2nd place** (at most 1 faster): **A2, B1**
- **3rd place** (at most 2 faster): **A3, B2, C1**, plus whichever of A2/B1 doesn't finish 2nd

Every other horse already has 3 or more horses known to be faster than it, so it can't be in the top 3.

## Race 7
Race **A2, A3, B1, B2, C1**: exactly 5 horses, so they fit in one race.
- The winner of race 7 is **2nd overall**.
- The runner-up of race 7 is **3rd overall**.

This always works. The race contains every horse that could be 2nd or 3rd, so whatever the result, the top two of race 7 are the overall 2nd and 3rd.

**Why 6 races can't be enough:** after race 6, A2 and B1 have never raced each other, so 2nd place can't be known yet.

## Final answer
| Goal | Races |
|---|---|
| 1st | 6 |
| 1st, 2nd, 3rd | **7** |
