---
sidebar_label: Apply a chunker for search
filename: apply_a_chunker_for_search.md
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import DownloadButton from '../downloadButton.js';


<!-- TABS -->
# Apply a chunker for search

:::note
Note that applying a chunker is ***not*** mandatory for search.
If your data is already chunked (e.g. short text snippets or audio) or if you
are searching through something like images, which can't be chunked, then this
won't be necessary.
:::


<Tabs>
    <TabItem value="Text" label="Text" default>
        ```python
        from superduperdb import objectmodel
        
        CHUNK_SIZE = 200
        
        @objectmodel(flatten=True, model_update_kwargs={'document_embedded': False})
        def chunker(text):
            text = text.split()
            chunks = [' '.join(text[i:i + CHUNK_SIZE]) for i in range(0, len(text), CHUNK_SIZE)]
            return chunks        
        ```
    </TabItem>
    <TabItem value="PDF" label="PDF" default>
        ```python
        !pip install -q "unstructured[pdf]"
        from superduperdb import objectmodel
        from unstructured.partition.pdf import partition_pdf
        import PyPDF2
        
        CHUNK_SIZE = 500
        
        @objectmodel(flatten=True, model_update_kwargs={'document_embedded': False})
        def chunker(pdf_file):
            elements = partition_pdf(pdf_file)
            text = '\n'.join([e.text for e in elements])
            chunks = [text[i:i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
            return chunks        
        ```
    </TabItem>
    <TabItem value="Video" label="Video" default>
        ```python
        !pip install opencv-python
        import cv2
        import tqdm
        from PIL import Image
        from superduperdb.ext.pillow import pil_image
        from superduperdb import objectmodel, Schema
        
        
        @objectmodel(
            flatten=True,
            model_update_kwargs={'document_embedded': False},
        )
        def chunker(video_file):
            # Set the sampling frequency for frames
            sample_freq = 10
            
            # Open the video file using OpenCV
            cap = cv2.VideoCapture(video_file)
            
            # Initialize variables
            frame_count = 0
            fps = cap.get(cv2.CAP_PROP_FPS)
            extracted_frames = []
            progress = tqdm.tqdm()
        
            # Iterate through video frames
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Get the current timestamp based on frame count and FPS
                current_timestamp = frame_count // fps
                
                # Sample frames based on the specified frequency
                if frame_count % sample_freq == 0:
                    extracted_frames.append({
                        'image': Image.fromarray(frame[:,:,::-1]),  # Convert BGR to RGB
                        'current_timestamp': current_timestamp,
                    })
                frame_count += 1
                progress.update(1)
            
            # Release resources
            cap.release()
            cv2.destroyAllWindows()
            
            # Return the list of extracted frames
            return extracted_frames        
        ```
    </TabItem>
    <TabItem value="Audio" label="Audio" default>
        ```python
        from superduperdb import objectmodel, Schema
        
        CHUNK_SIZE = 10  # in seconds
        
        @objectmodel(
            flatten=True,
            model_update_kwargs={'document_embedded': False},
            output_schema=Schema(identifier='output-schema', fields={'audio': datatype}),
        )
        def chunker(audio):
            chunks = []
            for i in range(0, len(audio), CHUNK_SIZE):
                chunks.append(audio[1][i: i + CHUNK_SIZE])
            return [(audio[0], chunk) for chunk in chunks]        
        ```
    </TabItem>
</Tabs>
Now we apply this chunker to the data by wrapping the chunker in `Listener`:

```python
from superduperdb import Listener

upstream_listener = Listener(
    model=chunker,
    select=select,
    key='x',
    uuid="chunk",
)

db.apply(upstream_listener)
```

<DownloadButton filename="apply_a_chunker_for_search.md" />
