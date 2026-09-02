Goal: Create a guessing puzzle game, identical to https://semantle.com/, except using epicure: (https://huggingface.co/Kaikaku/epicure-cooc, https://huggingface.co/Kaikaku/epicure-chem, https://huggingface.co/Kaikaku/epicure-core) as the engine for similarity

every day, new ingredient chosen, user tries to guess ingredient by inputting in textbox

Instead of just similarity, like in semantle, guesses should reveal:
1. score of how well it goes well with the target ingredient (using cooc)
2. score of how well the guess shares its flavor profile with the target ingredient (using chem)
3. score of how similar it is to the target ingredient (using core)

Each of these scores should show in a table on the page, with top scoring guesses for each (sorted from highest scoring to lowest)

measurements of cold -> tepid -> warm -> hot should be given when comparing  scores against nearest 10 neighbors for that measurement. Unlike semantle, no need to indicate rank

when correct, indicate to the user they're correct with a confetti effect and a winning popup showing how many guesses they took

Game name: epicuriosimi