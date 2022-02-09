TP1 - Analyse linguistique avec le framework NLTK.

Installation de la plateforme d’analyse linguistique NLTK.
	1. Installation de NLTK : 
		pip install --user -U nltk 
	2. Installation de Numpy : 
		pip install --user -U numpy 

I. Evaluation de l’analyse morpho-syntaxique de la plateforme NLTK.

	1. 
		python3 run_postagger_nltk.py wsj_0010_sample.txt wsj_0010_sample.txt.pos.nltk
		(Voir fichier : wsj_0010_sample.txt.pos.nltk)
	2. 
		python3 evaluate.py wsj_0010_sample.txt.pos.nltk wsj_0010_sample.pos.ref
			Word precision: 0.9454545454545454
			Word recall: 0.9454545454545454
			Tag precision: 0.9454545454545454
			Tag recall: 0.9454545454545454
			Word F-measure: 0.9454545454545454
			Tag F-measure: 0.9454545454545454
	(ps : Notre postagger écrit déja les tokens du fichier source (avec nltk) dans la forme que celui de référence.
	3. 
		a. Les deux lignes suivantes sont nécessaires pour convertir notre ficher (avec nltk) et aussi la référence.
			python3 run_use_universal.py POSTags_PTB_Universal_Linux.txt wsj_0010_sample.txt.pos.nltk wsj_0010_sample.txt.pos.univ.nltk
			python3 run_use_universal.py POSTags_PTB_Universal_Linux.txt wsj_0010_sample.pos.ref wsj_0010_sample.txt.pos.univ.ref
			(Voir fichier : wsj_0010_sample.txt.pos.univ.nltk & wsj_0010_sample.txt.pos.univ.ref)
		b. 
			python3 evaluate.py wsj_0010_sample.txt.pos.univ.nltk wsj_0010_sample.txt.pos.univ.ref
				Word precision: 0.9636363636363636
				Word recall: 0.9636363636363636
				Tag precision: 0.9636363636363636
				Tag recall: 0.9636363636363636
				Word F-measure: 0.9636363636363636
				Tag F-measure: 0.9636363636363636
		c. 
			On remarque que les étiquettes universelles améliorent la precision, le rappel et la F-mesure.
			Ce qui s'explique par la différence du nombre d'étiquettes, les étiquettes universelles étant 
				moins nombreuses que les étiquettes de Peen TreeBank cela limite les erreurs possibles.
			On en conclu que, pour plus "précision" dans l’analyse morpho-syntaxique, il est mieux d'utiliser
				les étiquettes universelles.
	
II. Utilisation de la plateforme NLTK pour l’analyse syntaxique.			

	1. Nous ne sommes pas sur d'avoir bien compris la consigne.
		Dans cette question, nous récupérons seulement les entités avec le tag "Compound".
			python3 run_parse_nltk.py wsj_0010_sample.txt wsj_0010_sample.txt.chk.nltk
			(Voir fichier : wsj_0010_sample.txt.chk.nltk)
	2. Nous n'avons pas utiliser de fichier déclaratif pour les structures syntaxiques. 
		Nous l'avons mis directement sous forme de grammaire.
			python3 run_parse_nltk2.py wsj_0010_sample.txt wsj_0010_sample.txt.chk.nltk2
			(Voir fichier : wsj_0010_sample.txt.chk.nltk2)

III. Utilisation de la plateforme NLTK pour l’extraction d’entités nommée.

	1.
		python3 extract_named_entity.py wsj_0010_sample.txt wsj_0010_sample.txt.ne.nltk
		(Voir fichier : wsj_0010_sample.txt.ne.nltk)
	2.
		python3 convert_named_entity_to_std_etiquettes.py wsj_0010_sample.txt.ne.nltk wsj_0010_sample.txt.ne.std.nltk
		(Voir fichier : wsj_0010_sample.txt.ne.std.nltk)
	3.
		python3 extract_named_entity.py formal-tst.NE.key.04oct95_sample.txt formal-tst.NE.key.04oct95_sample.txt.ne.nltk
		python3 convert_named_entity_to_std_etiquettes.py formal-tst.NE.key.04oct95_sample.txt.ne.nltk formal-tst.NE.key.04oct95_sample.txt.ne.std.nltk
		(Voir fichier : formal-tst.NE.key.04oct95_sample.txt.ne.std.nltk)
	