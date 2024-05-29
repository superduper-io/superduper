
# Vector search

:::note
Since vector-search is all-the-rage right now, 
here is the simplest possible iteration of semantic 
text-search with a `sentence_transformers` model, 
as an entrypoint to `superduperdb`.

Note that `superduperdb` is much-much more than vector-search
on text. Explore the docs to read about classical machine learning, 
computer vision, LLMs, fine-tuning and much much more!
:::


First let's get some data. These data are the markdown files 
of the very same documentation you are reading!
You can download other sample data-sets for testing `superduperdb`
by following [these lines of code](../reusable_snippets/get_useful_sample_data).

```python
import json
import requests 
r = requests.get('https://superduperdb-public-demo.s3.amazonaws.com/text.json')

with open('text.json', 'wb') as f:
    f.write(r.content)

with open('text.json', 'r') as f:
    data = json.load(f)        

print(data[0])
```

<details>
<summary>Outputs</summary>
<pre>
    ---
    sidebar_position: 5
    ---
    
    # Encoding data
    
    In AI, typical types of data are:
    
    - **Numbers** (integers, floats, etc.)
    - **Text**
    - **Images**
    - **Audio**
    - **Videos**
    - **...bespoke in house data**
    
    Most databases don't support any data other than numbers and text.
    SuperDuperDB enables the use of these more interesting data-types using the `Document` wrapper.
    
    ### `Document`
    
    The `Document` wrapper, wraps dictionaries, and is the container which is used whenever 
    data is exchanged with your database. That means inputs, and queries, wrap dictionaries 
    used with `Document` and also results are returned wrapped with `Document`.
    
    Whenever the `Document` contains data which is in need of specialized serialization,
    then the `Document` instance contains calls to `DataType` instances.
    
    ### `DataType`
    
    The [`DataType` class](../apply_api/datatype), allows users to create and encoder custom datatypes, by providing 
    their own encoder/decoder pairs.
    
    Here is an example of applying an `DataType` to add an image to a `Document`:
    
    ```python
    import pickle
    import PIL.Image
    from superduperdb import DataType, Document
    
    image = PIL.Image.open('my_image.jpg')
    
    my_image_encoder = DataType(
        identifier='my-pil',
        encoder=lambda x: pickle.dumps(x),
        decoder=lambda x: pickle.loads(x),
    )
    
    document = Document(\{'img': my_image_encoder(image)\})
    ```
    
    The bare-bones dictionary may be exposed with `.unpack()`:
    
    ```python
    \>\>\> document.unpack()
    \{'img': \<PIL.PngImagePlugin.PngImageFile image mode=P size=400x300\>\}
    ```
    
    By default, data encoded with `DataType` is saved in the database, but developers 
    may alternatively save data in the `db.artifact_store` instead. 
    
    This may be achiever by specifying the `encodable=...` parameter:
    
    ```python
    my_image_encoder = DataType(
        identifier='my-pil',
        encoder=lambda x: pickle.dumps(x),
        decoder=lambda x: pickle.loads(x),
        encodable='artifact',    # saves to disk/ db.artifact_store
        # encodable='lazy_artifact', # Just in time loading
    )
    ```
    
    The `encodable` specifies the type of the output of the `__call__` method, 
    which will be a subclass of `superduperdb.components.datatype._BaseEncodable`.
    These encodables become leaves in the tree defines by a `Document`.
    
    ### `Schema`
    
    A `Schema` allows developers to connect named fields of dictionaries 
    or columns of `pandas.DataFrame` objects with `DataType` instances.
    
    A `Schema` is used, in particular, for SQL databases/ tables, and for 
    models that return multiple outputs.
    
    Here is an example `Schema`, which is used together with text and image 
    fields:
    
    ```python
    s = Schema('my-schema', fields=\{'my-text': 'str', 'my-image': my_image_encoder\})
    ```
    

</pre>
</details>

Now we connect to SuperDuperDB, using MongoMock as a databackend.
Read more about connecting to SuperDuperDB [here](../core_api/connect) and
a semi-exhaustive list of supported data-backends for connecting [here](../reusable_snippets/connect_to_superduperdb).

```python
from superduperdb import superduper, Document

db = superduper('mongomock://test')

_ = db['documents'].insert_many([Document({'txt': txt}) for txt in data]).execute()
```

<details>
<summary>Outputs</summary>
<pre>
    2024-May-23 22:32:53.64| INFO     | Duncans-MBP.fritz.box| superduperdb.base.build:69   | Data Client is ready. mongomock.MongoClient('localhost', 27017)
    2024-May-23 22:32:53.66| INFO     | Duncans-MBP.fritz.box| superduperdb.base.build:42   | Connecting to Metadata Client with engine:  mongomock.MongoClient('localhost', 27017)
    2024-May-23 22:32:53.66| INFO     | Duncans-MBP.fritz.box| superduperdb.base.build:155  | Connecting to compute client: None
    2024-May-23 22:32:53.66| INFO     | Duncans-MBP.fritz.box| superduperdb.base.datalayer:85   | Building Data Layer
    2024-May-23 22:32:53.66| INFO     | Duncans-MBP.fritz.box| superduperdb.base.build:220  | Configuration: 
     +---------------+------------------+
    | Configuration |      Value       |
    +---------------+------------------+
    |  Data Backend | mongomock://test |
    +---------------+------------------+
    2024-May-23 22:32:53.67| INFO     | Duncans-MBP.fritz.box| superduperdb.backends.local.compute:37   | Submitting job. function:\<function callable_job at 0x1107caa20\>
    2024-May-23 22:32:53.68| SUCCESS  | Duncans-MBP.fritz.box| superduperdb.backends.local.compute:43   | Job submitted on \<superduperdb.backends.local.compute.LocalComputeBackend object at 0x15267d010\>.  function:\<function callable_job at 0x1107caa20\> future:03704b18-e98c-4eb8-ab48-d257105c3e6f

</pre>
</details>

```python
db.show()
```

<details>
<summary>Outputs</summary>
<pre>
    []
</pre>
</details>

We are going to make these data searchable by activating a [`Model`](../apply_api/model) instance 
to compute vectors for each item inserted to the `"documents"` collection.
For that we'll use the [sentence-transformers](https://sbert.net/) integration to `superduperdb`.
Read more about the `sentence_transformers` integration [here](../ai_integrations/sentence_transformers)
and [here](../../api/ext/sentence_transformers/).

```python

```

<details>
<summary>Outputs</summary>

</details>

```python
from superduperdb.ext.sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    identifier="test",
    predict_kwargs={"show_progress_bar": True},
    model="all-MiniLM-L6-v2",
    device="cpu",
    postprocess=lambda x: x.tolist(),
)
```

<details>
<summary>Outputs</summary>
<pre>
    /Users/dodo/.pyenv/versions/3.11.7/envs/superduperdb-3.11/lib/python3.11/site-packages/huggingface_hub/file_download.py:1132: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
      warnings.warn(

</pre>
<pre>
    2024-May-23 22:33:00.27| INFO     | Duncans-MBP.fritz.box| superduperdb.components.component:386  | Initializing SentenceTransformer : test
    2024-May-23 22:33:00.27| INFO     | Duncans-MBP.fritz.box| superduperdb.components.component:389  | Initialized  SentenceTransformer : test successfully

</pre>
<pre>
    Batches:   0%|          | 0/1 [00:00\<?, ?it/s]
</pre>
</details>

We can check that this model gives us what we want by evaluating an output 
on a single data-point. (Learn more about the various aspects of `Model` [here](../models/).)

```python
model.predict_one(data[0])
```

<details>
<summary>Outputs</summary>
<pre>
    Batches:   0%|          | 0/1 [00:00\<?, ?it/s]
</pre>
<pre>
    [-0.0728381797671318,
     -0.04369897395372391,
     -0.053990256041288376,
     0.05244452506303787,
     -0.023977573961019516,
     0.01649916172027588,
     -0.011447322554886341,
     0.061035461723804474,
     -0.07156683504581451,
     -0.021972885355353355,
     0.01267794519662857,
     0.018208766356110573,
     0.05270218849182129,
     -0.020327100530266762,
     -0.019956670701503754,
     0.027658769860863686,
     0.05226463824510574,
     -0.09045840799808502,
     -0.05595366284251213,
     -0.015193621627986431,
     0.11809872835874557,
     0.006927163805812597,
     -0.042815908789634705,
     0.020163120701909065,
     -0.007551214192062616,
     0.05370991304516792,
     -0.06269364058971405,
     -0.015371082350611687,
     0.07905995100736618,
     0.01635877788066864,
     0.013246661052107811,
     0.05565343424677849,
     0.01678791269659996,
     0.08823869377374649,
     -0.06329561769962311,
     0.018252376466989517,
     0.01689964346587658,
     -0.09000741690397263,
     -0.013926311396062374,
     -0.054565709084272385,
     0.09763795882463455,
     -0.045446526259183884,
     -0.11169185489416122,
     -0.01672297902405262,
     0.028883105143904686,
     0.02041822485625744,
     -0.07608168572187424,
     -0.03668771684169769,
     -0.03977571055293083,
     0.03618845343589783,
     -0.0918053463101387,
     0.029532095417380333,
     -0.04220665618777275,
     0.13082784414291382,
     0.024324564263224602,
     0.025249535217881203,
     -0.016180122271180153,
     0.010552441701292992,
     0.0027522461023181677,
     0.07488349825143814,
     0.010217947885394096,
     -0.005154070910066366,
     0.04516936093568802,
     -0.020390896126627922,
     0.039545465260744095,
     -0.031169062480330467,
     -0.04323659837245941,
     0.020132744684815407,
     0.0670941099524498,
     -0.08838536590337753,
     -0.005763655062764883,
     0.014565517194569111,
     -0.03434328734874725,
     0.08641394972801208,
     0.03842026740312576,
     -0.06397535651922226,
     -0.004498982336372137,
     -0.03862036392092705,
     0.009389184415340424,
     -0.06150598078966141,
     -0.018634818494319916,
     -0.04963228479027748,
     0.046070147305727005,
     0.07461931556463242,
     0.016484497115015984,
     -0.041531577706336975,
     0.07544152438640594,
     0.009718718007206917,
     -0.029345177114009857,
     0.009860241785645485,
     -0.01978706754744053,
     -0.1266753077507019,
     -0.006543521303683519,
     0.004957514349371195,
     -0.022630779072642326,
     0.062321994453668594,
     -0.008847227320075035,
     -0.009422101080417633,
     0.07500597834587097,
     -0.011071165092289448,
     -0.011291230097413063,
     -0.0023497703950852156,
     -0.0020577553659677505,
     -0.022909553721547127,
     -0.02039080671966076,
     -0.08629532903432846,
     0.035559117794036865,
     0.004795302636921406,
     -0.025927048176527023,
     -0.000661480997223407,
     -0.01712101511657238,
     -0.019804038107395172,
     -0.09941169619560242,
     -0.12973709404468536,
     -0.036208849400281906,
     0.01095140166580677,
     -0.10579997301101685,
     0.018861284479498863,
     -0.06653669476509094,
     -0.009016807191073895,
     0.01645195111632347,
     0.05936351791024208,
     0.024916797876358032,
     0.06697884202003479,
     0.06224494054913521,
     0.029584383592009544,
     -0.07033423334360123,
     2.664977201744624e-33,
     0.021844016388058662,
     -0.08870648592710495,
     -0.0011538445251062512,
     0.023276792839169502,
     -0.018942521885037422,
     0.008065970614552498,
     -0.03592826798558235,
     0.08716072887182236,
     0.02071245387196541,
     0.06679968535900116,
     -0.02447657659649849,
     0.0386064276099205,
     -0.058644849807024,
     0.05417194217443466,
     0.04741952195763588,
     0.03192991763353348,
     -0.07583042234182358,
     -0.016834404319524765,
     0.005513317883014679,
     0.03408630192279816,
     0.09274633228778839,
     0.03650207445025444,
     -0.009820879437029362,
     0.03678520396351814,
     0.04744667559862137,
     0.03139625862240791,
     -0.02660897560417652,
     -0.054728686809539795,
     -0.0004101162194274366,
     0.012437778525054455,
     -0.057767197489738464,
     -0.12133049219846725,
     0.004859662614762783,
     -0.005881409160792828,
     0.03496640920639038,
     0.0011129904305562377,
     -0.032958950847387314,
     -0.01912698708474636,
     -0.09516117721796036,
     0.01166975311934948,
     0.02697627805173397,
     0.04149679094552994,
     -0.038904909044504166,
     -0.07173115015029907,
     -0.03998439013957977,
     0.03461567685008049,
     0.056760404258966446,
     0.038543105125427246,
     -0.005076229106634855,
     -0.048972200602293015,
     -0.032644398510456085,
     0.04734884947538376,
     -0.028061121702194214,
     -0.015486026182770729,
     0.04073994979262352,
     -0.010933760553598404,
     0.07432980090379715,
     0.045219823718070984,
     0.061553847044706345,
     -0.04286878556013107,
     -0.04373219981789589,
     -0.030894780531525612,
     0.037015534937381744,
     -0.012399295344948769,
     -0.040280185639858246,
     0.018744098022580147,
     0.04238991066813469,
     0.0028010543901473284,
     0.11493761837482452,
     -0.01020615641027689,
     -0.05960821732878685,
     0.10087733715772629,
     -0.0005544194718822837,
     0.003897483227774501,
     -0.017415126785635948,
     0.021939443424344063,
     -0.023334739729762077,
     -0.1286034733057022,
     -0.05948842316865921,
     0.01876923255622387,
     -0.010775558650493622,
     -0.005998989101499319,
     -0.017639417201280594,
     0.02809220552444458,
     -0.05434253439307213,
     0.013654942624270916,
     -0.007518705911934376,
     -0.10503417998552322,
     -0.005824903957545757,
     -0.10465069860219955,
     0.053811464458703995,
     0.012696388177573681,
     -0.03567223250865936,
     -0.12682373821735382,
     -0.04431791231036186,
     -5.649626418775523e-33,
     -0.010820495896041393,
     -0.00802531372755766,
     -0.05365433543920517,
     0.03958006575703621,
     -0.02104414999485016,
     0.006130194291472435,
     0.04468188062310219,
     0.05036340653896332,
     -0.018140576779842377,
     -0.04300504922866821,
     0.012102029286324978,
     -0.00577476667240262,
     0.03385505825281143,
     -0.06575366109609604,
     -0.00653001619502902,
     0.016766566783189774,
     -0.12117733806371689,
     -0.09218579530715942,
     0.007316686678677797,
     -0.019426673650741577,
     -0.05662667751312256,
     0.0824657529592514,
     0.029016738757491112,
     0.047513313591480255,
     0.05799231678247452,
     -0.008996383287012577,
     -0.04977172240614891,
     0.03319057077169418,
     0.11511028558015823,
     0.02250896953046322,
     0.02120146155357361,
     -0.049932535737752914,
     -0.041500966995954514,
     -0.009317374788224697,
     -0.09659228473901749,
     -0.05510890483856201,
     0.06295066326856613,
     0.024173501878976822,
     -0.04577157646417618,
     0.024133509024977684,
     0.04559364914894104,
     0.021016940474510193,
     -0.049103744328022,
     0.024935618042945862,
     -0.05304615944623947,
     -0.014961606822907925,
     -0.09521036595106125,
     0.029579075053334236,
     0.025183551013469696,
     -0.08900482952594757,
     0.07622205466032028,
     -0.036385778337717056,
     -0.05705392360687256,
     -0.03871440514922142,
     0.011190380901098251,
     -0.046501439064741135,
     -0.025219706818461418,
     0.0001118649379350245,
     -0.04297145828604698,
     0.06217939034104347,
     0.04021172970533371,
     -0.07403939962387085,
     -0.0007105112308636308,
     0.0006416494725272059,
     -0.07840533554553986,
     -0.026061616837978363,
     -0.021549392491579056,
     -0.06263766437768936,
     -0.11086386442184448,
     -0.05587910860776901,
     0.07480043172836304,
     -0.07763925194740295,
     0.04992743954062462,
     0.06204086169600487,
     -0.0013184875715523958,
     -0.004204373806715012,
     -0.05604926869273186,
     -0.0030061916913837194,
     0.02281804382801056,
     0.0618956983089447,
     -0.046122197061777115,
     0.0020551434718072414,
     0.050125688314437866,
     0.08694882690906525,
     0.06670200824737549,
     0.018796533346176147,
     -0.010559462942183018,
     0.06277848035097122,
     -0.04749680310487747,
     -0.0014071549521759152,
     -0.08777493238449097,
     0.09142813831567764,
     -0.09544055908918381,
     0.09548325836658478,
     -0.01017127837985754,
     -5.976371397764524e-08,
     -0.07207749783992767,
     -0.018692996352910995,
     0.02441777102649212,
     0.047647666186094284,
     0.007122713141143322,
     -0.055901724845170975,
     -0.022228682413697243,
     0.08026605099439621,
     0.05604938790202141,
     -0.03505357354879379,
     0.06595908850431442,
     -0.02741447649896145,
     -0.1040404811501503,
     -0.013773254118859768,
     0.11995217949151993,
     0.00027782461256720126,
     0.07589304447174072,
     -0.009353208355605602,
     -0.013621047139167786,
     -0.03814826160669327,
     -0.03208579123020172,
     -0.04983912780880928,
     -0.0672062411904335,
     -0.08362551778554916,
     0.00817915890365839,
     0.011041522957384586,
     0.013109216466546059,
     0.13754235208034515,
     0.006957167759537697,
     -0.0294102281332016,
     0.011861572042107582,
     0.016042795032262802,
     0.10429029911756516,
     -0.0032936607021838427,
     0.02154575102031231,
     0.06281223148107529,
     0.03468304127454758,
     0.05810246244072914,
     -0.031500834971666336,
     0.014499562792479992,
     0.05990524962544441,
     -0.01979857124388218,
     -0.09960303455591202,
     0.0047220210544764996,
     0.07983221858739853,
     0.009491761215031147,
     0.06561334431171417,
     -0.007396489381790161,
     0.062069281935691833,
     0.05087302252650261,
     -0.0004922127700410783,
     -0.05793500691652298,
     0.03456997871398926,
     0.08377060294151306,
     0.03708452731370926,
     0.03597697988152504,
     -0.01678624376654625,
     -0.018676387146115303,
     0.06553706526756287,
     0.022750001400709152,
     0.015125676058232784,
     0.032285671681165695,
     0.03319930657744408,
     0.016521509736776352]
</pre>
</details>

Now that we've verified that this model works, we can "activate" it for 
vector-search by creating a [`VectorIndex`](../apply_api/vector_index).

```python
import pprint

vector_index = model.to_vector_index(select=db['documents'].find(), key='txt')

pprint.pprint(vector_index)
```

<details>
<summary>Outputs</summary>
<pre>
    VectorIndex(identifier='test:vector_index',
                uuid='acd20227-14e2-4cee-9507-f738315f5d42',
                indexing_listener=Listener(identifier='component/listener/test/b335fc9c-ad9e-4495-8c39-6894c5b4f842',
                                           uuid='b335fc9c-ad9e-4495-8c39-6894c5b4f842',
                                           key='txt',
                                           model=SentenceTransformer(preferred_devices=('cuda',
                                                                                        'mps',
                                                                                        'cpu'),
                                                                     device='cpu',
                                                                     identifier='test',
                                                                     uuid='11063ea2-4afa-4cab-8a55-21d0c7ad2900',
                                                                     signature='singleton',
                                                                     datatype=DataType(identifier='test/datatype',
                                                                                       uuid='e46268dc-5c88-48dd-8595-f774c35a8f09',
                                                                                       encoder=None,
                                                                                       decoder=None,
                                                                                       info=None,
                                                                                       shape=(384,),
                                                                                       directory=None,
                                                                                       encodable='native',
                                                                                       bytes_encoding=\<BytesEncoding.BYTES: 'Bytes'\>,
                                                                                       intermediate_type='bytes',
                                                                                       media_type=None),
                                                                     output_schema=None,
                                                                     flatten=False,
                                                                     model_update_kwargs=\{\},
                                                                     predict_kwargs=\{'show_progress_bar': True\},
                                                                     compute_kwargs=\{\},
                                                                     validation=None,
                                                                     metric_values=\{\},
                                                                     object=SentenceTransformer(
      (0): Transformer(\{'max_seq_length': 256, 'do_lower_case': False\}) with Transformer model: BertModel 
      (1): Pooling(\{'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True\})
      (2): Normalize()
    ),
                                                                     model='all-MiniLM-L6-v2',
                                                                     preprocess=None,
                                                                     postprocess=\<function \<lambda\> at 0x152658cc0\>),
                                           select=documents.find(),
                                           active=True,
                                           predict_kwargs=\{\}),
                compatible_listener=None,
                measure=\<VectorIndexMeasureType.cosine: 'cosine'\>,
                metric_values=\{\})

</pre>
</details>

You will see that the `VectorIndex` contains a [`Listener`](../apply_api/listener) instance.
This instance wraps the model, and configures it to compute outputs 
on data inserted to the `"documents"` collection with the key `"txt"`.

To activate this index, we now do:

```python
db.apply(vector_index)
```

<details>
<summary>Outputs</summary>
<pre>
    2024-May-23 22:33:06.79| INFO     | Duncans-MBP.fritz.box| superduperdb.components.component:386  | Initializing DataType : dill_lazy
    2024-May-23 22:33:06.79| INFO     | Duncans-MBP.fritz.box| superduperdb.components.component:389  | Initialized  DataType : dill_lazy successfully
    2024-May-23 22:33:08.38| INFO     | Duncans-MBP.fritz.box| superduperdb.components.component:386  | Initializing DataType : dill
    2024-May-23 22:33:08.38| INFO     | Duncans-MBP.fritz.box| superduperdb.components.component:389  | Initialized  DataType : dill successfully
    2024-May-23 22:33:08.42| INFO     | Duncans-MBP.fritz.box| superduperdb.backends.local.compute:37   | Submitting job. function:\<function method_job at 0x1107caac0\>

</pre>
<pre>
    204it [00:00, 142844.41it/s]
</pre>
<pre>
    2024-May-23 22:33:08.55| INFO     | Duncans-MBP.fritz.box| superduperdb.components.component:386  | Initializing SentenceTransformer : test
    2024-May-23 22:33:08.55| INFO     | Duncans-MBP.fritz.box| superduperdb.components.component:389  | Initialized  SentenceTransformer : test successfully

</pre>
<pre>
    

</pre>
<pre>
    Batches:   0%|          | 0/7 [00:00\<?, ?it/s]
</pre>
<pre>
    2024-May-23 22:33:12.78| INFO     | Duncans-MBP.fritz.box| superduperdb.components.model:783  | Adding 204 model outputs to `db`
    2024-May-23 22:33:12.89| WARNING  | Duncans-MBP.fritz.box| superduperdb.backends.mongodb.query:254  | Some delete ids are not executed , hence halting execution Please note the partially executed operations wont trigger any `model/listeners` unless CDC is active.
    2024-May-23 22:33:12.89| SUCCESS  | Duncans-MBP.fritz.box| superduperdb.backends.local.compute:43   | Job submitted on \<superduperdb.backends.local.compute.LocalComputeBackend object at 0x15267d010\>.  function:\<function method_job at 0x1107caac0\> future:3598065c-0bfb-4d94-9b25-6e7e82c09bd0
    2024-May-23 22:33:12.90| INFO     | Duncans-MBP.fritz.box| superduperdb.backends.local.compute:37   | Submitting job. function:\<function callable_job at 0x1107caa20\>
    2024-May-23 22:33:12.98| INFO     | Duncans-MBP.fritz.box| superduperdb.base.datalayer:170  | Loading vectors of vector-index: 'test:vector_index'
    2024-May-23 22:33:12.98| INFO     | Duncans-MBP.fritz.box| superduperdb.base.datalayer:180  | documents.find(documents[0], documents[1])

</pre>
<pre>
    Loading vectors into vector-table...: 204it [00:00, 3148.10it/s]
</pre>
<pre>
    2024-May-23 22:33:13.05| SUCCESS  | Duncans-MBP.fritz.box| superduperdb.backends.local.compute:43   | Job submitted on \<superduperdb.backends.local.compute.LocalComputeBackend object at 0x15267d010\>.  function:\<function callable_job at 0x1107caa20\> future:c355caeb-daab-4712-a269-6bfca8da2c09

</pre>
<pre>
    

</pre>
<pre>
    ([\<superduperdb.jobs.job.ComponentJob at 0x28d6f95d0\>,
      \<superduperdb.jobs.job.FunctionJob at 0x28d757850\>],
     VectorIndex(identifier='test:vector_index', uuid='acd20227-14e2-4cee-9507-f738315f5d42', indexing_listener=Listener(identifier='component/listener/test/b335fc9c-ad9e-4495-8c39-6894c5b4f842', uuid='b335fc9c-ad9e-4495-8c39-6894c5b4f842', key='txt', model=SentenceTransformer(preferred_devices=('cuda', 'mps', 'cpu'), device='cpu', identifier='test', uuid='11063ea2-4afa-4cab-8a55-21d0c7ad2900', signature='singleton', datatype=DataType(identifier='test/datatype', uuid='e46268dc-5c88-48dd-8595-f774c35a8f09', encoder=None, decoder=None, info=None, shape=(384,), directory=None, encodable='native', bytes_encoding=\<BytesEncoding.BYTES: 'Bytes'\>, intermediate_type='bytes', media_type=None), output_schema=None, flatten=False, model_update_kwargs=\{\}, predict_kwargs=\{'show_progress_bar': True\}, compute_kwargs=\{\}, validation=None, metric_values=\{\}, object=SentenceTransformer(
       (0): Transformer(\{'max_seq_length': 256, 'do_lower_case': False\}) with Transformer model: BertModel 
       (1): Pooling(\{'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True\})
       (2): Normalize()
     ), model='all-MiniLM-L6-v2', preprocess=None, postprocess=\<function \<lambda\> at 0x152658cc0\>), select=documents.find(), active=True, predict_kwargs=\{\}), compatible_listener=None, measure=\<VectorIndexMeasureType.cosine: 'cosine'\>, metric_values=\{\}))
</pre>
</details>

The `db.apply` command is a universal command for activating AI components in SuperDuperDB.

You will now see lots of output - the model-outputs/ vectors are computed 
and the various parts of the `VectorIndex` are registered in the system.

You can verify this with:

```python
db.show()
```

<details>
<summary>Outputs</summary>
<pre>
    [\{'identifier': 'test', 'type_id': 'model'\},
     \{'identifier': 'component/listener/test/b335fc9c-ad9e-4495-8c39-6894c5b4f842',
      'type_id': 'listener'\},
     \{'identifier': 'test:vector_index', 'type_id': 'vector_index'\}]
</pre>
</details>

```python
db['documents'].find_one().execute().unpack()
```

<details>
<summary>Outputs</summary>
<pre>
    \{'txt': "---\nsidebar_position: 5\n---\n\n# Encoding data\n\nIn AI, typical types of data are:\n\n- **Numbers** (integers, floats, etc.)\n- **Text**\n- **Images**\n- **Audio**\n- **Videos**\n- **...bespoke in house data**\n\nMost databases don't support any data other than numbers and text.\nSuperDuperDB enables the use of these more interesting data-types using the `Document` wrapper.\n\n### `Document`\n\nThe `Document` wrapper, wraps dictionaries, and is the container which is used whenever \ndata is exchanged with your database. That means inputs, and queries, wrap dictionaries \nused with `Document` and also results are returned wrapped with `Document`.\n\nWhenever the `Document` contains data which is in need of specialized serialization,\nthen the `Document` instance contains calls to `DataType` instances.\n\n### `DataType`\n\nThe [`DataType` class](../apply_api/datatype), allows users to create and encoder custom datatypes, by providing \ntheir own encoder/decoder pairs.\n\nHere is an example of applying an `DataType` to add an image to a `Document`:\n\n```python\nimport pickle\nimport PIL.Image\nfrom superduperdb import DataType, Document\n\nimage = PIL.Image.open('my_image.jpg')\n\nmy_image_encoder = DataType(\n    identifier='my-pil',\n    encoder=lambda x: pickle.dumps(x),\n    decoder=lambda x: pickle.loads(x),\n)\n\ndocument = Document(\{'img': my_image_encoder(image)\})\n```\n\nThe bare-bones dictionary may be exposed with `.unpack()`:\n\n```python\n\>\>\> document.unpack()\n\{'img': \<PIL.PngImagePlugin.PngImageFile image mode=P size=400x300\>\}\n```\n\nBy default, data encoded with `DataType` is saved in the database, but developers \nmay alternatively save data in the `db.artifact_store` instead. \n\nThis may be achiever by specifying the `encodable=...` parameter:\n\n```python\nmy_image_encoder = DataType(\n    identifier='my-pil',\n    encoder=lambda x: pickle.dumps(x),\n    decoder=lambda x: pickle.loads(x),\n    encodable='artifact',    # saves to disk/ db.artifact_store\n    # encodable='lazy_artifact', # Just in time loading\n)\n```\n\nThe `encodable` specifies the type of the output of the `__call__` method, \nwhich will be a subclass of `superduperdb.components.datatype._BaseEncodable`.\nThese encodables become leaves in the tree defines by a `Document`.\n\n### `Schema`\n\nA `Schema` allows developers to connect named fields of dictionaries \nor columns of `pandas.DataFrame` objects with `DataType` instances.\n\nA `Schema` is used, in particular, for SQL databases/ tables, and for \nmodels that return multiple outputs.\n\nHere is an example `Schema`, which is used together with text and image \nfields:\n\n```python\ns = Schema('my-schema', fields=\{'my-text': 'str', 'my-image': my_image_encoder\})\n```\n",
     '_fold': 'train',
     '_id': ObjectId('664fa7f5df381fe5ebf38405'),
     '_outputs': \{'b335fc9c-ad9e-4495-8c39-6894c5b4f842': [-0.0728381797671318,
       -0.04369895160198212,
       -0.053990304470062256,
       0.05244451016187668,
       -0.023977596312761307,
       0.016499122604727745,
       -0.011447325348854065,
       0.061035484075546265,
       -0.07156682759523392,
       -0.021972879767417908,
       0.012677934020757675,
       0.018208758905529976,
       0.052702222019433975,
       -0.020327096804976463,
       -0.01995668187737465,
       0.027658754959702492,
       0.05226461961865425,
       -0.09045842289924622,
       -0.05595369264483452,
       -0.015193603932857513,
       0.11809875071048737,
       0.006927188020199537,
       -0.042815886437892914,
       0.02016316168010235,
       -0.007551214657723904,
       0.05370989069342613,
       -0.06269364058971405,
       -0.015371100045740604,
       0.07905995845794678,
       0.01635879836976528,
       0.01324666291475296,
       0.05565342679619789,
       0.016787931323051453,
       0.08823872357606888,
       -0.06329561024904251,
       0.018252374604344368,
       0.016899660229682922,
       -0.0900074765086174,
       -0.013926304876804352,
       -0.05456570163369179,
       0.09763795137405396,
       -0.04544650763273239,
       -0.11169182509183884,
       -0.016722947359085083,
       0.028883112594485283,
       0.02041824720799923,
       -0.07608170062303543,
       -0.0366877056658268,
       -0.03977571055293083,
       0.036188457161188126,
       -0.09180538356304169,
       0.02953210100531578,
       -0.04220666363835335,
       0.130827859044075,
       0.024324607104063034,
       0.025249570608139038,
       -0.01618010364472866,
       0.010552453808486462,
       0.0027521972078830004,
       0.07488350570201874,
       0.010217934846878052,
       -0.005154080223292112,
       0.04516935348510742,
       -0.020390905439853668,
       0.039545439183712006,
       -0.03116907924413681,
       -0.04323665052652359,
       0.020132753998041153,
       0.0670941025018692,
       -0.08838535100221634,
       -0.0057636769488453865,
       0.014565511606633663,
       -0.034343305975198746,
       0.08641396462917328,
       0.03842025622725487,
       -0.06397533416748047,
       -0.004498984199017286,
       -0.038620349019765854,
       0.009389190003275871,
       -0.0615059956908226,
       -0.018634818494319916,
       -0.04963228479027748,
       0.0460701584815979,
       0.07461929321289062,
       0.016484474763274193,
       -0.04153159260749817,
       0.07544155418872833,
       0.009718707762658596,
       -0.02934517152607441,
       0.009860233403742313,
       -0.019787028431892395,
       -0.1266753375530243,
       -0.006543517112731934,
       0.0049575152806937695,
       -0.022630779072642326,
       0.06232202798128128,
       -0.00884722638875246,
       -0.0094221206381917,
       0.07500597089529037,
       -0.011071158573031425,
       -0.011291255243122578,
       -0.0023497689981013536,
       -0.0020577521063387394,
       -0.022909540683031082,
       -0.020390814170241356,
       -0.08629532158374786,
       0.035559121519327164,
       0.004795318003743887,
       -0.025927070528268814,
       -0.0006614814046770334,
       -0.017121002078056335,
       -0.019804026931524277,
       -0.09941168129444122,
       -0.12973710894584656,
       -0.03620882332324982,
       0.010951397940516472,
       -0.10579998791217804,
       0.018861234188079834,
       -0.06653666496276855,
       -0.009016799740493298,
       0.01645198091864586,
       0.05936354771256447,
       0.02491680160164833,
       0.06697887927293777,
       0.062244962900877,
       0.02958441898226738,
       -0.07033420354127884,
       2.6649770180736317e-33,
       0.021844014525413513,
       -0.08870648592710495,
       -0.0011538179824128747,
       0.0232767965644598,
       -0.01894250698387623,
       0.008065932430326939,
       -0.035928282886743546,
       0.08716070652008057,
       0.020712479948997498,
       0.06679967790842056,
       -0.02447659522294998,
       0.03860645368695259,
       -0.0586448572576046,
       0.054171912372112274,
       0.04741951450705528,
       0.03192995861172676,
       -0.07583040744066238,
       -0.01683441549539566,
       0.0055133323185145855,
       0.034086331725120544,
       0.09274634718894958,
       0.03650207072496414,
       -0.009820892475545406,
       0.036785200238227844,
       0.047446656972169876,
       0.031396280974149704,
       -0.02660900540649891,
       -0.05472869798541069,
       -0.0004101111553609371,
       0.01243777945637703,
       -0.05776720494031906,
       -0.12133051455020905,
       0.004859668668359518,
       -0.005881412420421839,
       0.03496639057993889,
       0.0011130021885037422,
       -0.03295896574854851,
       -0.019126981496810913,
       -0.09516119956970215,
       0.011669756844639778,
       0.026976292952895164,
       0.04149681329727173,
       -0.03890489786863327,
       -0.07173115760087967,
       -0.039984408766031265,
       0.03461568057537079,
       0.056760385632514954,
       0.03854312747716904,
       -0.005076217465102673,
       -0.048972200602293015,
       -0.03264437988400459,
       0.047348879277706146,
       -0.028061090037226677,
       -0.015485992655158043,
       0.04073994606733322,
       -0.010933739133179188,
       0.07432981580495834,
       0.04521976783871651,
       0.06155385449528694,
       -0.04286882281303406,
       -0.04373219609260559,
       -0.03089478425681591,
       0.037015557289123535,
       -0.012399279512465,
       -0.04028019309043884,
       0.018744099885225296,
       0.04238991439342499,
       0.002801078837364912,
       0.11493764072656631,
       -0.01020615454763174,
       -0.05960826203227043,
       0.10087732970714569,
       -0.00055444345343858,
       0.003897454123944044,
       -0.017415115609765053,
       0.02193945087492466,
       -0.02333473414182663,
       -0.1286035031080246,
       -0.0594884529709816,
       0.01876922883093357,
       -0.0107755521312356,
       -0.0059989625588059425,
       -0.017639396712183952,
       0.02809220924973488,
       -0.05434252694249153,
       0.01365494355559349,
       -0.0075187087059021,
       -0.10503418743610382,
       -0.00582492258399725,
       -0.10465067625045776,
       0.053811490535736084,
       0.012696387246251106,
       -0.03567224740982056,
       -0.12682373821735382,
       -0.04431792348623276,
       -5.649626418775523e-33,
       -0.010820521041750908,
       -0.008025307208299637,
       -0.05365429446101189,
       0.03958004713058472,
       -0.02104412391781807,
       0.0061301738023757935,
       0.044681861996650696,
       0.050363361835479736,
       -0.01814057119190693,
       -0.0430050753057003,
       0.012102004140615463,
       -0.005774796940386295,
       0.033855050802230835,
       -0.06575366854667664,
       -0.00653000408783555,
       0.016766533255577087,
       -0.12117733061313629,
       -0.09218578785657883,
       0.007316680159419775,
       -0.01942664571106434,
       -0.05662669613957405,
       0.08246578276157379,
       0.02901674434542656,
       0.04751332476735115,
       0.05799226835370064,
       -0.00899638794362545,
       -0.04977171868085861,
       0.033190544694662094,
       0.11511028558015823,
       0.02250894159078598,
       0.021201487630605698,
       -0.0499325729906559,
       -0.0415009967982769,
       -0.009317407384514809,
       -0.09659233689308167,
       -0.05510890111327171,
       0.06295064091682434,
       0.024173470214009285,
       -0.04577154666185379,
       0.024133525788784027,
       0.045593682676553726,
       0.02101696841418743,
       -0.049103744328022,
       0.024935608729720116,
       -0.053046178072690964,
       -0.014961596578359604,
       -0.09521038830280304,
       0.029579076915979385,
       0.02518356405198574,
       -0.08900485187768936,
       0.07622209191322327,
       -0.03638580068945885,
       -0.05705391988158226,
       -0.03871438279747963,
       0.011190400458872318,
       -0.04650144279003143,
       -0.025219738483428955,
       0.00011186233314219862,
       -0.04297143965959549,
       0.06217937543988228,
       0.040211718529462814,
       -0.07403940707445145,
       -0.0007105701370164752,
       0.0006416687392629683,
       -0.07840534299612045,
       -0.026061605662107468,
       -0.02154943160712719,
       -0.06263765692710876,
       -0.11086387932300568,
       -0.05587908253073692,
       0.07480042427778244,
       -0.07763926684856415,
       0.049927398562431335,
       0.06204086169600487,
       -0.001318484079092741,
       -0.004204366356134415,
       -0.05604930222034454,
       -0.0030061937868595123,
       0.02281801961362362,
       0.06189575046300888,
       -0.046122193336486816,
       0.0020551353227347136,
       0.05012568086385727,
       0.08694884926080704,
       0.06670202314853668,
       0.018796497955918312,
       -0.01055945549160242,
       0.06277844309806824,
       -0.04749682545661926,
       -0.0014071010518819094,
       -0.08777494728565216,
       0.09142817556858063,
       -0.09544060379266739,
       0.09548324346542358,
       -0.010171260684728622,
       -5.976371397764524e-08,
       -0.07207749783992767,
       -0.0186929851770401,
       0.02441776543855667,
       0.047647684812545776,
       0.007122725248336792,
       -0.05590169504284859,
       -0.022228669375181198,
       0.080266073346138,
       0.056049395352602005,
       -0.03505353629589081,
       0.06595905870199203,
       -0.02741449698805809,
       -0.1040404662489891,
       -0.013773255050182343,
       0.11995211988687515,
       0.0002778216148726642,
       0.07589299976825714,
       -0.009353214874863625,
       -0.013621056452393532,
       -0.03814827278256416,
       -0.0320858396589756,
       -0.04983909800648689,
       -0.06720622628927231,
       -0.08362554013729095,
       0.008179157972335815,
       0.01104153972119093,
       0.013109265826642513,
       0.13754235208034515,
       0.0069571868516504765,
       -0.02941022627055645,
       0.011861592531204224,
       0.01604282297194004,
       0.10429032146930695,
       -0.003293645801022649,
       0.021545739844441414,
       0.06281221657991409,
       0.03468310087919235,
       0.05810248851776123,
       -0.031500861048698425,
       0.014499560929834843,
       0.05990522727370262,
       -0.01979857124388218,
       -0.09960301965475082,
       0.0047220224514603615,
       0.07983223348855972,
       0.009491737931966782,
       0.06561332941055298,
       -0.007396463770419359,
       0.062069281935691833,
       0.050873052328825,
       -0.0004922244697809219,
       -0.05793503299355507,
       0.034569934010505676,
       0.08377060294151306,
       0.037084512412548065,
       0.03597693890333176,
       -0.0167862419039011,
       -0.018676359206438065,
       0.06553705036640167,
       0.022750040516257286,
       0.015125693753361702,
       0.032285649329423904,
       0.03319932520389557,
       0.016521470621228218]\}\}
</pre>
</details>

To "use" the `VectorIndex` we can execute a vector-search query:

```python
query = db['documents'].like({'txt': 'Tell me about vector-search'}, vector_index=vector_index.identifier, n=3).find()
cursor = query.execute()
```

<details>
<summary>Outputs</summary>
<pre>
    2024-May-23 22:33:16.62| INFO     | Duncans-MBP.fritz.box| superduperdb.base.datalayer:1095 | \{\}

</pre>
<pre>
    Batches:   0%|          | 0/1 [00:00\<?, ?it/s]
</pre>
</details>

This query will return a cursor of [`Document`](../fundamentals/document) instances.
To obtain the raw dictionaries, call the `.unpack()` command:

```python
for r in cursor:
    print('=' * 100)
    print(r.unpack()['txt'])
    print('=' * 100)
```

<details>
<summary>Outputs</summary>
<pre>
    ====================================================================================================
    ---
    sidebar_position: 7
    ---
    
    # Vector-search
    
    SuperDuperDB allows users to implement vector-search in their database by either 
    using in-database functionality, or via a sidecar implementation with `lance` and `FastAPI`.
    
    ## Philosophy
    
    In `superduperdb`, from a user point-of-view vector-search isn't a completely different beast than other ways of 
    using the system:
    
    - The vector-preparation is exactly the same as preparing outputs with any model, 
      with the special difference that the outputs are vectors, arrays or tensors.
    - Vector-searches are just another type of database query which happen to use 
      the stored vectors.
    
    ## Algorithm
    
    Here is a schematic of how vector-search works:
    
    ![](/img/vector-search.png)
    
    ## Explanation
    
    A vector-search query has the schematic form:
    
    ```python
    table_or_collection
        .like(Document(\<dict-to-search-with\>))      # the operand is vectorized using registered models
        .filter_results(*args, **kwargs)            # the results of vector-search are filtered
    ```
    
    ```python
    table_or_collection
        .filter_results(*args, **kwargs)            # the results of vector-search are filtered
        .like(Document(\<dict-to-search-with\>))      # the operand is vectorized using registered models
    ```
    
    ...or
    
    The type of such a query is a `CompoundSelect`. It's 2 parts are the vector-search part (`like`) and the 
    filtering part (`select`).
    
    In the first case, the operand of `like` is dispatched to a **model**, which converts this into a **vector**.
    The **vector** is compared to previously saved outputs of the same or a paired **model** (multi-modal).
    The most similar `ids` are retrieved. The `select` part of the query is then transformed to 
    a similar query which searches within the retrieved `ids`. The full set of results are returned
    to the client.
    
    Read [here](../walkthrough/vector_search.md) about setting up and detailed usage of vector-search.
    
    ====================================================================================================
    ====================================================================================================
    # Vector search queries
    
    Vector search queries are built with the `.like` operator.
    This allows developers to combine standard database with vector-search queries.
    The philosophy is that developers do not need to convert their inputs 
    into vector's themselves. Rather, this is taken care by the specified 
    [`VectorIndex` component](../apply_api/vector_index).
    
    The basic schematic for vector-search queries is:
    
    ```python
    table_or_collection
        .like(Document(\<dict-to-search-with\>), vector_index='\<my-vector-index\>')      # the operand is vectorized using registered models
        .filter_results(*args, **kwargs)            # the results of vector-search are filtered
    ```
    
    ***or...***
    
    ```python
    table_or_collection
        .filter_results(*args, **kwargs)            # the results of vector-search are filtered
        .like(Document(\<dict-to-search-with\>),
              vector_index='\<my-vector-index\>')      # the operand is vectorized using registered models
    ```
    
    ## MongoDB
    
    ```python
    from superduperdb.ext.pillow import pil_image
    from superduperdb import Document
    
    my_image = PIL.Image.open('test/material/data/test_image.png')
    
    q = my_collection.find(\{'brand': 'Nike'\}).like(Document(\{'img': pil_image(my_image)\}), 
                                                   vector_index='\<my-vector-index\>')
    
    results = db.execute(q)
    ```
    
    ## SQL
    
    ```python
    t = db.load('table', 'my-table')
    t.filter(t.brand == 'Nike').like(Document(\{'img': pil_image(my_image)\}))
    
    results = db.execute(q)
    ```
    
    
    ====================================================================================================
    ====================================================================================================
    # Sidecar vector-comparison integration
    
    For databases which don't have their own vector-search implementation, `superduperdb` offers 
    2 integrations:
    
    - In memory vector-search
    - Lance vector-search
    
    To configure these, add one of the following options to your configuration:
    
    ```yaml
    cluster:
      vector_search:
        type: in_memory|lance
    ```
    
    ***or***
    
    ```bash
    export SUPERDUPER_CLUSTER_VECTOR_SEARCH_TYPE='in_memory|lance'
    ```
    
    In this case, whenever a developer executes a vector-search query including `.like`, 
    execution of the similarity and sorting computations of vectors is outsourced to 
    a sidecar implementation which is managed by `superduperdb`.
    ====================================================================================================

</pre>
</details>

You should see that the documents returned are relevant to the `like` part of the 
query.

Learn more about building queries with `superduperdb` [here](../execute_api/overview.md).
