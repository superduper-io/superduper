# OpenAI vector search

In this tutorial we'll implement vector-search in SuperDuperDB using OpenAI vector-embeddings.
For this you'll need an `OPENAI_API_KEY`, which you can obtain on [openai's website](https://openai.com/blog/openai-api). You'll then set this as an environment variable:


```python
import os

# Get one here https://openai.com/blog/openai-api
os.environ['OPENAI_API_KEY'] = '<YOUR-API-KEY>'
```


```python
import pymongo

from superduperdb.ext.openai.model import OpenAIEmbedding
from superduperdb.db.mongodb.query import Collection
from superduperdb import superduper
from superduperdb.container.listener import Listener
from superduperdb.container.vector_index import VectorIndex
```

In order to access SuperDuperDB, we use the `Datalayer` class. This is obtained by simply wrapping 
your MongoDB database connection, using `pymongo`. We'll be adding data and activating models on the `wikipedia`
collection:


```python
db = pymongo.MongoClient().documents
db = superduper(db)

collection = Collection(name='wikipedia')
```

We'll use a tiny excerpt out of wikipedia to test the functionality. Needless to say, since the `Datalayer` uses
MongoDB and LanceDB, the vector-search solution may be taken to large scale without a problem:


```python
data = [
  {
    "title": "Anarchism",
    "abstract": "Anarchism is a political philosophy and movement that is skeptical of all justifications for authority and seeks to abolish the institutions they claim maintain unnecessary coercion and hierarchy, typically including, though not necessarily limited to, the state and capitalism. Anarchism advocates for the replacement of the state with stateless societies or other forms of free associations."
  },
  {
    "title": "Albedo",
    "abstract": "Albedo (; ) is the measure of the diffuse reflection of solar radiation out of the total solar radiation and measured on a scale from 0, corresponding to a black body that absorbs all incident radiation, to 1, corresponding to a body that reflects all incident radiation."
  },
  {
    "title": "Achilles",
    "abstract": "In Greek mythology, Achilles ( ) or Achilleus ( Accessed 5 May 2017. the latter being the dative of the former. The name grew more popular, even becoming common soon after the seventh century BCEpigraphical database gives 476 matches for Ἀχιλ-.The earliest ones: Corinth 7th c. BC, Delphi 530 BC, Attica and Elis 5th c. BC. and was also turned into the female form Ἀχιλλεία (Achilleía), attested in Attica in the fourth century BC (IG II² 1617) and, in the form Achillia, on a stele in Halicarnassus as the name of a female gladiator fighting an \"Amazon\"."
  },
  {
    "title": "An American in Paris",
    "abstract": "An American in Paris is a jazz-influenced orchestral piece by American composer George Gershwin first performed in 1928. It was inspired by the time that Gershwin had spent in Paris and evokes the sights and energy of the French capital during the Années folles."
  },
  {
    "title": "Academy Award for Best Production Design",
    "abstract": "The Academy Award for Best Production Design recognizes achievement for art direction in film. The category's original name was Best Art Direction, but was changed to its current name in 2012 for the 85th Academy Awards."
  },
  {
    "title": "Animalia (book)",
    "abstract": "Animalia is an illustrated children's book by Graeme Base. It was originally published in 1986, followed by a tenth anniversary edition in 1996, and a 25th anniversary edition in 2012."
  },
  {
    "title": "International Atomic Time",
    "abstract": "International Atomic Time (abbreviated TAI, from its French name Temps atomique 1975) is a high-precision atomic coordinate time standard based on the notional passage of proper time on Earth's geoid. It is a continuous scale of time, without leap seconds, and it is the principal realisation of Terrestrial Time (with a fixed offset of epoch)."
  },
  {
    "title": "Altruism",
    "abstract": "Altruism is the principle and moral practice of concern for happiness of other human beings or other animals, resulting in a quality of life both material and spiritual. It is a traditional virtue in many cultures and a core aspect of various religious and secular worldviews."
  },
  {
    "title": "Anthropology",
    "abstract": "Anthropology is the scientific study of humanity, concerned with human behavior, human biology, cultures, societies, and linguistics, in both the present and past, including past human species. Social anthropology studies patterns of behaviour, while cultural anthropology studies cultural meaning, including norms and values."
  },
  {
    "title": "Agricultural science",
    "abstract": "Agricultural science (or agriscience for short Meaning &amp; Definition for UK English |encyclopedia=Oxford Dictionary of English |url=https://www.lexico."
  },
  {
    "title": "Alchemy",
    "abstract": "Alchemy (from Arabic: al-kīmiyā; from Ancient Greek: khumeía);  Definition of alchemy in English by Oxford Dictionaries|website=Oxford Dictionaries  English|access-date=30 September 2018}} is an ancient branch of natural philosophy, a philosophical and protoscientific tradition that was historically practiced in China, India, the Muslim world, and Europe. In its Western form, alchemy is first attested in a number of pseudepigraphical texts written in Greco-Roman Egypt during the first few centuries AD."
  },
  {
    "title": "Astronomer",
    "abstract": "An astronomer is a scientist in the field of astronomy who focuses their studies on a specific question or field outside the scope of Earth. They observe astronomical objects such as stars, planets, moons, comets and galaxies – in either observational (by analyzing the data) or theoretical astronomy."
  },
  {
    "title": "Animation",
    "abstract": "Animation is a method in which figures are manipulated to appear as moving images. In traditional animation, images are drawn or painted by hand on transparent celluloid sheets to be photographed and exhibited on film."
  },
  {
    "title": "Austroasiatic languages",
    "abstract": "The Austroasiatic languages , , also known as Mon–KhmerBradley (2012) notes, MK in the wider sense including the Munda languages of eastern South Asia is also known as Austroasiatic. , are a large language family in  Mainland Southeast Asia and South Asia."
  },
  {
    "title": "Arithmetic mean",
    "abstract": "In mathematics and statistics, the arithmetic mean ( ) or arithmetic average, or just the mean or the average (when the context is clear), is the sum of a collection of numbers divided by the count of numbers in the collection. The collection is often a set of results of an experiment or an observational study, or frequently a set of results from a survey."
  },
  {
    "title": "American Football Conference",
    "abstract": "The American Football Conference (AFC) is one of the two conferences of the National Football League (NFL), the highest professional level of American football in the United States. This conference currently contains 16 teams organized into 4 divisions, as does its counterpart, the National Football Conference (NFC)."
  },
  {
    "title": "Animal Farm",
    "abstract": "Animal Farm is a beast fable, in form of satirical allegorical novella, by George Orwell, first published in England on 17 August 1945. It tells the story of a group of farm animals who rebel against their human farmer, hoping to create a society where the animals can be equal, free, and happy."
  },
  {
    "title": "Agriculture",
    "abstract": "Agriculture or farming is the practice of cultivating plants and livestock. |year=1999 |publisher=International Labour Organization |isbn=978-92-2-111517-5 |page=77 |access-date=13 September 2010 |url-status=live |archive-url=https://web."
  },
  {
    "title": "Analysis of variance",
    "abstract": "Analysis of variance (ANOVA) is a collection of statistical models and their associated estimation procedures (such as the \"variation\" among and between groups) used to analyze the differences among means. ANOVA was developed by the statistician Ronald Fisher."
  },
  {
    "title": "Appellate procedure in the United States",
    "abstract": "United States appellate procedure involves the rules and regulations for filing appeals in state courts and federal courts. The nature of an appeal can vary greatly depending on the type of case and the rules of the court in the jurisdiction where the case was prosecuted."
  },
  {
    "title": "Answer (law)",
    "abstract": "In law, an answer was originally a solemn assertion in opposition to someone or something, and thus generally any counter-statement or defense, a reply to a question or response, or objection, or a correct solution of a problem."
  },
  {
    "title": "Arraignment",
    "abstract": "Arraignment is a formal reading of a criminal charging document in the presence of the defendant, to inform them of the charges against them. In response to arraignment, the accused is expected to enter a plea."
  },
  {
    "title": "America the Beautiful",
    "abstract": "\"America the Beautiful\" is a patriotic American song. Its lyrics were written by Katharine Lee Bates and its music was composed by church organist and choirmaster Samuel A."
  },
  {
    "title": "Asphalt",
    "abstract": "Asphalt, also known as bitumen (, ), is a sticky, black, highly viscous liquid or semi-solid form of petroleum. It may be found in natural deposits or may be a refined product, and is classed as a pitch."
  },
  {
    "title": "Argument (disambiguation)",
    "abstract": "In logic and philosophy, an argument is an attempt to persuade someone of something, or give evidence or reasons for accepting a particular conclusion."
  },
  {
    "title": "Astronaut",
    "abstract": "An astronaut (from the Ancient Greek  (), meaning 'star', and  (), meaning 'sailor') is a person trained, equipped, and deployed by a human spaceflight program to serve as a commander or crew member aboard a spacecraft. Although generally reserved for professional space travelers, the term is sometimes applied to anyone who travels into space, including scientists, politicians, journalists, and tourists."
  },
  {
    "title": "A Modest Proposal",
    "abstract": "A Modest Proposal For preventing the Children of Poor People From being a Burthen to Their Parents or Country, and For making them Beneficial to the Publick, commonly referred to as A Modest Proposal, is a Juvenalian satirical essay written and published anonymously by Jonathan Swift in 1729. The essay suggests that the impoverished Irish might ease their economic troubles by selling their children as food to rich gentlemen and ladies."
  },
  {
    "title": "Alkali metal",
    "abstract": "The alkali metals consist of the chemical elements lithium (Li), sodium (Na), potassium (K),The symbols Na and K for sodium and potassium are derived from their Latin names, natrium and kalium; these are still the origins of the names for the elements in some languages, such as German and Russian. rubidium (Rb), caesium (Cs),."
  },
  {
    "title": "Alphabet",
    "abstract": "An alphabet is a standardized set of basic written symbols or graphemes (called letters) that represent the phonemes of certain spoken languages. Not all writing systems represent language in this way; in a syllabary, each character represents a syllable, for instance, and logographic systems use characters to represent words, morphemes, or other semantic units."
  },
  {
    "title": "Atomic number",
    "abstract": "thumb|right|300px|The Rutherford–Bohr model of the [[hydrogen atom ( 1}}) or a hydrogen-like ion (). In this model it is an essential feature that the photon energy (or frequency) of the electromagnetic radiation emitted (shown) when an electron jumps from one orbital to another be proportional to the mathematical square of atomic charge ()."
  },
  {
    "title": "Affirming the consequent",
    "abstract": "Affirming the consequent, sometimes called converse error, fallacy of the converse, or confusion of necessity and sufficiency, is a formal fallacy of taking a true conditional statement (e.g."
  },
  {
    "title": "Ambiguity",
    "abstract": "Ambiguity is a type of meaning in which a phrase, statement or resolution is not explicitly defined, making several interpretations [A common aspect of ambiguity is uncertainty]. It is thus an attribute of any idea or statement whose [[intention|intended meaning cannot be definitively resolved according to a rule or process with a finite number of steps."
  },
  {
    "title": "Abel",
    "abstract": "Abel;  Héḇel, in pausa  Hā́ḇel;  Hábel; }} is a Biblical figure in the Book of Genesis within Abrahamic religions. He was the younger brother of Cain, and the younger son of Adam and Eve, the first couple in Biblical history."
  },
  {
    "title": "Adobe",
    "abstract": "Adobe ( ; ) is a building material made from earth and organic materials,  is Spanish for mudbrick. In some English-speaking regions of Spanish heritage, such as the Southwestern United States, the term is used to refer to any kind of earthen construction, or various architectural styles like Pueblo Revival or Territorial Revival."
  },
  {
    "title": "Adventure",
    "abstract": "An adventure is an exciting experience or undertaking that is typically bold, sometimes risky. Adventures may be activities with danger such as traveling, exploring, skydiving, mountain climbing, scuba diving, river rafting, or other extreme sports."
  },
  {
    "title": "Articles of Confederation",
    "abstract": "The Articles of Confederation and Perpetual Union was an agreement among the 13 original states of the United States of America that served as its first frame of government. It was approved after much debate (between July 1776 and November 1777) by the Second Continental Congress on November 15, 1777, and sent to the states for ratification."
  },
  {
    "title": "Asia Minor (disambiguation)",
    "abstract": "Asia Minor is an alternative name for Anatolia, the westernmost protrusion of Asia, comprising the majority of the Republic of Turkey."
  },
  {
    "title": "Demographics of Angola",
    "abstract": "This article is about the demographic features of the population of Angola, including population density, ethnicity, education level, health of the populace, economic status, religious affiliations and other aspects of the population."
  },
  {
    "title": "Politics of Angola",
    "abstract": "The Angolan government is composed of three branches of government: executive, legislative and judicial. For decades, political power has been concentrated in the presidency with the People's Movement for the Liberation of Angola."
  },
  {
    "title": "Algorithms (journal)",
    "abstract": "Algorithms is a monthly peer-reviewed open-access scientific journal of mathematics, covering design, analysis, and experiments on algorithms. The journal is published by MDPI and was established in 2008."
  },
  {
    "title": "Amateur astronomy",
    "abstract": "thumb|right|250px|Amateur astronomers listen the night sky during the [[Perseids|Perseid meteor shower.]]"
  },
  {
    "title": "Art",
    "abstract": "Art is a diverse range of human activity, and resulting product, that involves creative or imaginative talent expressive of technical proficiency, beauty, emotional power, or conceptual ideas. Definition of Conceptual Art by Oxford Dictionary on Lexico."
  },
  {
    "title": "Abstract (law)",
    "abstract": "In law, an abstract is a brief statement that contains the most important points of a long legal document or of several related legal papers."
  },
  {
    "title": "Ampere",
    "abstract": "The ampere (, ; symbol: A), often shortened to amp,SI supports only the use of symbols and deprecates the use of abbreviations for units. is the SI base unit of electric current."
  },
  {
    "title": "Algorithm",
    "abstract": "thumb|right| [[Flowchart of an algorithm (Euclid's algorithm) for calculating the greatest common divisor (g.c."
  },
  {
    "title": "Anthophyta",
    "abstract": "The anthophytes are a grouping of plant taxa bearing flower-like reproductive structures. They were formerly thought  to be a clade comprising plants bearing flower-like structures."
  },
  {
    "title": "Mouthwash",
    "abstract": "Mouthwash, mouth rinse, oral rinse, or mouth bath is a liquid which is held in the mouth passively or swilled around the mouth by contraction of the perioral muscles and/or movement of the head, and may be gargled, where the head is tilted back and the liquid bubbled at the back of the mouth."
  },
  {
    "title": "Asteroid",
    "abstract": "An asteroid is a minor planet of the inner Solar System. Sizes and shapes of asteroids vary significantly, ranging from 1-meter rocks to dwarf planets almost 1000 km in diameter; they are metallic or rocky bodies with no atmosphere."
  },
  {
    "title": "Allocution",
    "abstract": "An allocution, or allocutus, is a formal statement made to the court by the defendant who has been found guilty prior to being sentenced. It is part of the criminal procedure in some jurisdictions using common law."
  },
  {
    "title": "Affidavit",
    "abstract": "An  ( ; Medieval Latin for \"he has declared under oath\") is a written statement voluntarily made by an affiant or deponent under an oath or affirmation which is administered by a person who is authorized to do so by law. Such a statement is witnessed as to the authenticity of the affiant's signature by a taker of oaths, such as a notary public or commissioner of oaths."
  },
  {
    "title": "Anime",
    "abstract": "|lead=yes}} is a Japanese term for  animation. Outside of Japan and in English, anime refers specifically to animation produced in Japan."
  },
  {
    "title": "Axiom of choice",
    "abstract": "In mathematics, the axiom of choice, or AC, is an axiom of set theory equivalent to the statement that a Cartesian product of a collection of non-empty sets is non-empty. Informally put, the axiom of choice says that given any collection of bins, each containing at least one object, it is possible to construct a set by arbitrarily choosing one object from each bin, even if the collection is infinite."
  },
  {
    "title": "A Clockwork Orange (novel)",
    "abstract": "A Clockwork Orange is a dystopian satirical black comedy novel by English writer Anthony Burgess, published in 1962. It is set in a near-future society that has a youth subculture of extreme violence."
  },
  {
    "title": "Amsterdam",
    "abstract": "| image_caption            = From top down, left to right: Keizersgracht, canal in the Centrum borough, the Royal Concertgebouw and Rijksmuseum"
  },
  {
    "title": "Museum of Work",
    "abstract": "The Museum of Work (Arbetets museum) is a museum located in Norrköping, Sweden. The museum is located in the Strykjärn (Clothes iron), a former weaving mill in the old industrial area on the Motala ström river in the city centre of Norrköping."
  },
  {
    "title": "Aircraft",
    "abstract": "An aircraft is a vehicle or machine that is able to fly by gaining support from the air. It counters the force of gravity by using either static lift or by using the dynamic lift of an airfoil, or in a few cases the downward thrust from jet engines."
  },
  {
    "title": "Motor neuron disease",
    "abstract": "Motor neuron diseases or motor neurone diseases (MNDs) are a group of rare neurodegenerative disorders that selectively affect motor neurons, the cells which control voluntary muscles of the body. They include amyotrophic lateral sclerosis (ALS), progressive bulbar palsy (PBP), pseudobulbar palsy, progressive muscular atrophy (PMA), primary lateral sclerosis (PLS), spinal muscular atrophy (SMA) and monomelic amyotrophy (MMA), as well as some rarer variants resembling ALS."
  },
  {
    "title": "Abjad",
    "abstract": "An abjad (, ; also abgad) is a writing system in which only consonants are represented, leaving vowel sounds to be inferred by the reader. This contrasts with other alphabets, which provide graphemes for both consonants and vowels."
  },
  {
    "title": "Abugida",
    "abstract": "thumb|300px|Comparison of various abugidas descended from [[Brahmi script. Meaning: May Śiva protect those who take delight in the language of the gods."
  },
  {
    "title": "Agnosticism",
    "abstract": "Agnosticism is the view or belief that the existence of God, of the divine or the supernatural is unknown or unknowable. (page 56 in 1967 edition) Another definition provided is the view that \"human reason is incapable of providing sufficient rational grounds to justify either the belief that God exists or the belief that God does not exist."
  },
  {
    "title": "Argon",
    "abstract": "Argon is a chemical element with the symbol Ar and atomic number 18. It is in group 18 of the periodic table and is a noble gas."
  },
  {
    "title": "Arsenic",
    "abstract": "Arsenic is a chemical element with the symbol As and atomic number 33. Arsenic occurs in many minerals, usually in combination with sulfur and metals, but also as a pure elemental crystal."
  },
  {
    "title": "Actinium",
    "abstract": "Actinium is a chemical element with the symbol Ac and atomic number 89. It was first isolated by Friedrich Oskar Giesel in 1902, who gave it the name emanium; the element got its name by being wrongly identified with a substance André-Louis Debierne found in 1899 and called actinium."
  },
  {
    "title": "Americium",
    "abstract": "Americium is a synthetic radioactive chemical element with the symbol Am and atomic number 95. It is a transuranic member of the actinide series, in the periodic table located under the lanthanide element europium, and thus by analogy was named after the Americas."
  },
  {
    "title": "Astatine",
    "abstract": "Astatine is a chemical element with the symbol At and atomic number 85. It is the rarest naturally occurring element in the Earth's crust, occurring only as the decay product of various heavier elements."
  },
  {
    "title": "Atom",
    "abstract": "An atom is the smallest unit of ordinary matter that forms a chemical element.McSween Jr, Harry, and Gary Huss."
  },
  {
    "title": "Aluminium",
    "abstract": "Aluminium (aluminum in American and Canadian English) is a chemical element with the symbol Al and atomic number 13. Aluminium has a density lower than those of other common metals, at approximately one third that of steel."
  },
  {
    "title": "Advanced Chemistry",
    "abstract": "Advanced Chemistry is a German hip hop group from Heidelberg, a scenic city in Baden-Württemberg, South Germany. Advanced Chemistry was founded in 1987 by Toni L, Linguist, Gee-One, DJ Mike MD (Mike Dippon) and MC Torch."
  },
  {
    "title": "Archipelago",
    "abstract": "An archipelago ( ), sometimes called an island group or island chain, is a chain, cluster, or collection of islands, or sometimes a sea containing a small number of scattered islands."
  },
  {
    "title": "Angst",
    "abstract": "Angst is fear or anxiety (anguish is its Latinate equivalent, and the words anxious and anxiety are of similar origin). The dictionary definition for angst is a feeling of anxiety, apprehension, or insecurity."
  },
  {
    "title": "Anxiety",
    "abstract": "Anxiety is an emotion which is characterized by an unpleasant state of inner [and it includes subjectively unpleasant feelings of dread over anticipated] events. It is often accompanied by nervous behavior such as pacing back and forth, [[Somatic anxiety|somatic complaints, and rumination."
  },
  {
    "title": "Axiom",
    "abstract": "An axiom, postulate, or assumption is a statement that is taken to be true, to serve as a premise or starting point for further reasoning and arguments. The word comes from the Ancient Greek word  (), meaning 'that which is thought worthy or fit' or 'that which commends itself as evident'."
  },
  {
    "title": "Alpha",
    "abstract": "Alpha  (uppercase , lowercase ; , álpha, or ) is the first letter of the Greek alphabet. In the system of Greek numerals, it has a value of one."
  },
  {
    "title": "Apiaceae",
    "abstract": "Apiaceae or Umbelliferae is a family of mostly aromatic flowering plants named after the type genus Apium and commonly known as the celery, carrot or parsley family, or simply as umbellifers. It is the 16th-largest family of flowering plants, with more than 3,700 species in 434 generaStevens, P."
  },
  {
    "title": "Axon",
    "abstract": "An axon (from Greek ἄξων áxōn, axis), or nerve fiber (or nerve fibre: see spelling differences), is a long, slender projection of a nerve cell, or neuron, in vertebrates, that typically conducts electrical impulses known as action potentials away from the nerve cell body. The function of the axon is to transmit information to different neurons, muscles, and glands."
  },
  {
    "title": "American shot",
    "abstract": "\"American shot\" or \"cowboy shot\" is a translation of a phrase from French film criticism, plan américain, and refers to a medium-long (\"knee\") film shot of a group of characters, who are arranged so that all are visible to the camera. The usual arrangement is for the actors to stand in an irregular line from one side of the screen to the other, with the actors at the end coming forward a little and standing more in profile than the others."
  },
  {
    "title": "Acute disseminated encephalomyelitis",
    "abstract": "Acute disseminated encephalomyelitis (ADEM), or acute demyelinating encephalomyelitis, is a rare autoimmune disease marked by a sudden, widespread attack of inflammation in the brain and spinal cord. As well as causing the brain and spinal cord to become inflamed, ADEM also attacks the nerves of the central nervous system and damages their myelin insulation, which, as a result, destroys the white matter."
  },
  {
    "title": "Ataxia",
    "abstract": "Ataxia is a neurological sign consisting of lack of voluntary coordination of muscle movements that can include gait abnormality, speech changes, and abnormalities in eye movements. Ataxia is a clinical manifestation indicating dysfunction of the parts of the nervous system that coordinate movement, such as the cerebellum."
  },
  {
    "title": "Applied ethics",
    "abstract": "Applied ethics refers to the practical application of moral considerations. It is ethics with respect to real-world actions and their moral considerations in the areas of private and public life, the professions, health, technology, law, and leadership."
  },
  {
    "title": "Miss Marple",
    "abstract": "Miss Marple is a fictional character in Agatha Christie's crime novels and short stories. Jane Marple lives in the village of St."
  },
  {
    "title": "Aaron",
    "abstract": "According to the Abrahamic religions, Aaron ′aharon, , Greek (Septuagint): [often called Aaron the priest ().|group=\"note\"}} ( or ;  ’Ahărōn) was a prophet], [[high priest, and the elder brother of Moses."
  },
  {
    "title": "Alcohol (chemistry)",
    "abstract": "In chemistry, an alcohol is a type of organic compound that carries at least one hydroxyl functional group (−OH) bound to a saturated carbon atom. The term alcohol originally referred to the primary alcohol ethanol (ethyl alcohol), which is used as a drug and is the main alcohol present in alcoholic drinks. An important class of alcohols, of which methanol and ethanol are the simplest examples, includes all compounds which conform to the general formula . Simple monoalcohols that are the subject of this article include primary (), secondary () and tertiary () alcohols."
  },
  {
    "title": "Algebraically closed field",
    "abstract": "In mathematics, a field  is algebraically closed if every non-constant polynomial in  (the univariate polynomial ring with coefficients in ) has a root in ."
  },
  {
    "title": "Aspect ratio",
    "abstract": "The aspect ratio of a geometric shape is the ratio of its sizes in different dimensions. For example, the aspect ratio of a rectangle is the ratio of its longer side to its shorter side—the ratio of width to height, when the rectangle is oriented as a \"landscape\"."
  },
  {
    "title": "Auto racing",
    "abstract": "Auto racing (also known as car racing, motor racing, or automobile racing) is a motorsport involving the racing of automobiles for competition."
  },
  {
    "title": "Anarcho-capitalism",
    "abstract": "Anarcho-capitalism (or, colloquially, ancap) Definition of ANCAP by Oxford Dictionary on Lexico.com also meaning of ANCAP |url=https://www."
  },
  {
    "title": "Aristophanes",
    "abstract": "| footnotes        = † Although many artists' renderings of Aristophanes portray him with flowing curly hair, several jests in his plays indicate that he may have been prematurely bald."
  },
  {
    "title": "Austrian School",
    "abstract": "The Austrian School is a heterodox school of economic thought that advocates strict adherence to methodological individualism, the concept that social phenomena result exclusively from the motivations and actions of individuals. Austrians school theorists hold that economic theory should be exclusively derived from basic principles of human action."
  },
  {
    "title": "Abatement",
    "abstract": "Abatement refers generally to a lessening, diminution, reduction, or moderation; specifically, it may refer to:"
  },
  {
    "title": "Amateur",
    "abstract": "An amateur (; ; ) is generally considered a person who pursues an avocation independent from their source of income. Amateurs and their pursuits are also described as popular, informal, self-taught, user-generated, DIY, and hobbyist."
  },
  {
    "title": "All Souls' Day",
    "abstract": "|litcolor = Black, where it is traditionGeneral Instruction of the Roman Missal, 346 (otherwise violet or purple)General Instruction of the Roman Missal, 346"
  },
  {
    "title": "Algorithms for calculating variance",
    "abstract": "Algorithms for calculating variance play a major role in computational statistics. A key difficulty in the design of good algorithms for this problem is that formulas for the variance may involve sums of squares, which can lead to numerical instability as well as to arithmetic overflow when dealing with large values."
  },
  {
    "title": "Politics of Antigua and Barbuda",
    "abstract": "The politics of Antigua and Barbuda takes place in a framework of a unitary parliamentary representative democratic monarchy, wherein the Sovereign of Antigua and Barbuda is the head of state, appointing a Governor-General to act as vice-regal representative in the nation. A Prime Minister is appointed by the Governor-General as the head of government, and of a multi-party system; the Prime Minister advises the Governor-General on the appointment of a Council of Ministers."
  },
  {
    "title": "Telecommunications in Antigua and Barbuda",
    "abstract": "Telecommunications in Antigua and Barbuda are via media in the telecommunications industry. This article is about communications systems in Antigua and Barbuda."
  },
  {
    "title": "Antisemitism",
    "abstract": "Antisemitism (also spelled anti-semitism or anti-Semitism) is hostility to, prejudice towards, or discrimination against Jews.See, for example:"
  },
  {
    "title": "Foreign relations of Azerbaijan",
    "abstract": "The Republic of Azerbaijan is a member of the United Nations, the Non-Aligned Movement, the Organization for Security and Cooperation in Europe, NATO's Partnership for Peace, the Euro-Atlantic Partnership Council, the World Health Organization, the European Bank for Reconstruction and Development; the Council of Europe, CFE Treaty, the Community of Democracies; the International Monetary Fund; and the World Bank."
  },
  {
    "title": "Politics of Armenia",
    "abstract": "The politics of Armenia take place in the framework of the parliamentary representative democratic republic of Armenia, whereby the President of Armenia is the head of state and the Prime Minister of Armenia the head of government, and of a multi-party system. Executive power is exercised by the President and the Government."
  },
  {
    "title": "Foreign relations of Armenia",
    "abstract": "Since its independence, Armenia has maintained a policy of complementarism by trying to have positive and friendly relations with Iran, Russia, and the West, including the United States and the European Union.– \"Armenian Foreign Policy Between Russia, Iran And U."
  },
  {
    "title": "Demographics of American Samoa",
    "abstract": "This article is about the demographics of American Samoa, including population density, ethnicity, education level, health of the populace, economic status, religious affiliations and other aspects of the population. American Samoa is an unincorporated territory of the United States located in the South Pacific Ocean."
  },
  {
    "title": "Analysis",
    "abstract": "Analysis is the process of breaking a complex topic or substance into smaller parts in order to gain a better understanding of it. The technique has been applied in the study of mathematics and logic since before Aristotle (384–322 B."
  }
]
```

Creating a vector-index in SuperDuperDB involves two things:

- Creating a model which is used to compute vectors (in this case `OpenAIEmbedding`)
- Daemonizing this model on a key (`Listener`), so that when new data are added, these are vectorized using the key

This may be done in multiple steps in or one command, as below:


```python
db.add(
    VectorIndex(
        identifier='my-index',
        indexing_listener=Listener(
            model=OpenAIEmbedding(model='text-embedding-ada-002'),
            key='abstract',
            select=Collection(name='wikipedia').find(),
        ),
    )
)

```

We can verify the components which have been activated in the database with `db.show`:


```python
print(db.show('listener'))
print(db.show('model'))
print(db.show('vector_index'))
```

You'll see now, that as data are added to the database, the model springs into action, vectorizing these
documents, and adding the vectors back to the original documents:


```python
from superduperdb.container.document import Document

data = [Document(r) for r in data]

db.execute(collection.insert_many(data))
```

We can verify that the vectors are in the documents:


```python
db.execute(collection.find_one())
```

Now we can use the vector-index to search via meaning through the wikipedia abstracts:


```python
cur = db.execute(
    Collection(name='wikipedia')
        .like({'abstract': 'philosophers'}, n=10, vector_index='my-index')
)

for r in cur:
    print(r['title'])
```

The upside of using a standard database as the databackend, is that we can combine vector-search
with standard filtering, to get a hybrid search:


```python
cur = db.execute(
    Collection(name='wikipedia')
        .like({'abstract': 'philosophers'}, n=100, vector_index='my-index')
        .find({'title': {'$regex': 'eth'}})
)

for r in cur:
    print(r['title'])
```
