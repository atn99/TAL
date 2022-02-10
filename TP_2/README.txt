TP2 - Analyse linguistique avec la plateforme Stanford CoreNLP.


I. Evaluation de l’outil de désambiguïsation morphosyntaxique de l’université de Stanford 

	a. 
		cd stanford-postagger-2018-10-16/
		./stanford-postagger.sh models/english-left3words-distsim.tagger ../wsj_0010_sample.txt > ../wsj_0010_sample.txt.pos.stanford
		cd ..

		(Voir fichier : wsj_0010_sample.txt.pos.stanford)

	b. 
		python3 convert_ref_into_standford.py wsj_0010_sample.pos.ref wsj_0010_sample.pos.stanford.ref

		(Voir fichier : wsj_0010_sample.pos.stanford.ref)
	c.

		python3 evaluate.py wsj_0010_sample.txt.pos.stanford wsj_0010_sample.pos.stanford.ref

				Word precision: 0.990909090909091
                                Word recall: 0.990909090909091
                                Tag precision: 0.9727272727272728
                                Tag recall: 0.9727272727272728
                                Word F-measure: 0.990909090909091
                                Tag F-measure: 0.9727272727272728

 
	d. 
		Les deux lignes suivantes sont nécessaires pour convertir notre ficher à la fois le fichier obtenu avec l'outil de stanford et le fichier de référence.
			python3 run_use_universal.py POSTags_PTB_Universal.txt wsj_0010_sample.txt.pos.stanford wsj_0010_sample.txt.pos.univ.stanford
			python3 run_use_universal.py POSTags_PTB_Universal.txt wsj_0010_sample.pos.stanford.ref wsj_0010_sample.txt.pos.univ.ref
			(Voir fichier : wsj_0010_sample.txt.pos.univ.stanford & wsj_0010_sample.txt.pos.univ.ref)
		

	e.
		python3 evaluate.py wsj_0010_sample.txt.pos.univ.stanford wsj_0010_sample.txt.pos.univ.ref

				Word precision: 0.990909090909091
				Word recall: 0.990909090909091
                        	Tag precision: 0.9727272727272728
                        	Tag recall: 0.9727272727272728
                        	Word F-measure: 0.990909090909091
                        	Tag F-measure: 0.9727272727272728



	f.
		Bien que lors du TP précédent nous ayons conclu que l'utilisation des étiquettes universelles permettait de réduire les erreurs d'évaluation du modèle, nous observons ici
		que l'utilisation de ces étiquettes n'impacte en rien les valeurs de précision de rappel et de F-mesure.
		On peut donc conclure que l'outil de désambiguïsation morphosyntaxique de l’université de Stanford est bien plus performant que celui de nltk. Ainsi, le fait de convertir
		en étiquettes universelles n'améliore pas la précision, car le résultat est déjà extrêmement précis.