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
        ```
    </TabItem>
    <TabItem value="Image" label="Image" default>
        ```python
        from IPython.display import display
        
        def visualize(item, source):
            display(item)        # item is a PIL.Image        
        ```
    </TabItem>
    <TabItem value="Audio" label="Audio" default>
        ```python
        from IPython.display import Audio, display
        
        def visualize(item, source):
            display(Audio(item[1], fs=item[0]))        
        ```
    </TabItem>
    <TabItem value="PDF" label="PDF" default>
        ```python
        from IPython.display import IFrame, display
        
        def visualize(item, source):
            display(IFrame(item))        
        ```
    </TabItem>
    <TabItem value="Video" label="Video" default>
        ```python
        from IPython.display import display, HTML
        
        timestamp = 0     # increment to the frame you want to start at
        
        # Create HTML code for the video player with a specified source and controls
        video_html = f"""
        <video width="640" height="480" controls>
            <source src="{video['video'].uri}" type="video/mp4">
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
        ```
    </TabItem>
</Tabs>
