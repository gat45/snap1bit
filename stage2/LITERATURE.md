# Croisement littérature — décisions Stage 2

Ce document sépare les faits mesurés du projet des conséquences tirées des
sources externes. Il ne convertit pas une hypothèse de performance en résultat.

## Format et préparation côté hôte

La documentation officielle du backend Snapdragon de llama.cpp recommande de
pré-calculer côté hôte les formes, strides, conversions de scale et layouts de
tuiles, puis de ne transférer vers VTCM que les données nécessaires. Elle
recommande aussi de privilégier DMA/VTCM plutôt que des accès scalaires dans
VTCM. Cela soutient un layout Q1 compact, mais impose que son permutation
bit→lanes soit explicitement définie et testée; le format dense ne peut pas
être passé au lecteur Q4_0 existant sans expansion.

Source: https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/snapdragon/developer.md

## Bande passante: hypothèse à mesurer, pas garantie

Réduire les octets de poids peut aider le decode, mais le gain dépend du régime
réel (bande passante, expansion HVX, DMA, coût de dispatch). Une tuile 192-o
ne donne un facteur trois que si les 128 octets de bits denses sont lus une
fois: un format répliqué à 512 octets supprimerait ce bénéfice. La littérature
sur l'inférence LLM confirme que la bande passante est un facteur majeur, sans
prédire la vitesse de ce kernel précis.

Source: https://arxiv.org/abs/2503.18869

## Qualité Q1: le kernel n'est pas le modèle

BitNet obtient des résultats à 1 bit avec un entraînement et une recette de
quantification adaptés. Les méthodes de quantification post-entraînement
extrême, elles, signalent une dégradation importante quand le budget de bits
descend. Cela concorde avec la PPL Nanbeige mesurée ici: un chemin DSP
bit-exact ne corrige pas une PTQ Q1_0 de mauvaise qualité.

Conséquence: Q1_0 reste expérimental, opt-in et validé par qualité sur chaque
modèle/tensor; le Stage 2 doit être évalué à qualité fixée contre Q4.

Sources: https://arxiv.org/abs/2402.17764 (BitNet),
https://arxiv.org/abs/2401.06118 (AQLM).

## Ce que les sources ne prouvent pas

- Elles ne valident ni le mapping HVX du prototype ni l'absence d'accès hors
  limites.
- Elles ne prédisent pas 20 t/s sur OnePlus 15.
- Elles ne rendent pas la PTQ Q1_0 naïve utilisable.

Les seuls critères de validation restent le simulateur de layout, les tests
HTP, la comparaison CPU et les benchmarks thermiquement contrôlés.
