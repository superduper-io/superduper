---
sidebar_label: Visualize Results
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

<!-- TABS -->
# Visualize Results


<Tabs>
    <TabItem value="Text" label="Text" default>
        ```python
        from IPython.display import Markdown, display
        
        def visualize(item, source):
            display(Markdown(item))
            
        def show(results, output_key, get_original_callable=None):
            for result in results:
                source = None
                if '_source' in result:
                    
                    source = get_original_callable(result['_source'])
                visualize(result[output_key], source)        
        ```
    </TabItem>
    <TabItem value="Image" label="Image" default>
        ```python
        from IPython.display import display
        
        def visualize(item, source):
            display(item)        # item is a PIL.Image
        
        def show(results, output_key, get_original_callable=None):
            for result in results:
                source = None
                if '_source' in result:
                    source = get_original_callable(result['_source'])
                visualize(result[output_key], source)        
        ```
    </TabItem>
    <TabItem value="Audio" label="Audio" default>
        ```python
        from IPython.display import Audio, display
        
        def visualize(item, source):
            display(Audio(item[1], fs=item[0]))
        
        def show(results, output_key, get_original_callable=None):
            for result in results:
                source = None
                if '_source' in result:
                    
                    source = get_original_callable(result['_source'])
                visualize(result[output_key], source)        
        ```
    </TabItem>
    <TabItem value="PDF" label="PDF" default>
        ```python
        from IPython.display import IFrame, display
        
        def visualize(item, source):
            display(item)
        
        
        def show(results, output_key, get_original_callable=None):
            for result in results:
                source = None
                if '_source' in result:
                    
                    source = get_original_callable(result['_source'])
                visualize(result[output_key], source)        
        ```
    </TabItem>
    <TabItem value="Video" label="Video" default>
        ```python
        from IPython.display import display, HTML
        
        def visualize(uri, source):
            timestamp = source    # increment to the frame you want to start at
            
            # Create HTML code for the video player with a specified source and controls
            video_html = f"""
            <video width="640" height="480" controls>
                <source src="{uri}" type="video/mp4">
            </video>
            <script>
                // Get the video element
                var video = document.querySelector('video');
                
                // Set the current time of the video to the specified timestamp
                video.currentTime = {timestamp};
                
                // Play the video automatically
                video.play();
            </script>
            """
            
            display(HTML(video_html))
        
        
        def show(results, output_key, get_original_callable=None):
            # show only the first video
            for result in results:
                result = result[output_key]
                timestamp = result['current_timestamp']
                source = result['_source']
                uri = get_original_callable(source)['x']
                visualize(uri, timestamp)
                break        
        ```
    </TabItem>
</Tabs>
