import os
import shutil
from superduper import Model, logging
from superduper.components.vector_index import VectorIndex


def rematch(texts, answer, n=5):
    texts_words = [set(t.lower().split()) for t in texts]
    answer_words = set(answer.lower().split())
    scores = [len(t & answer_words) / len(answer_words) for t in texts_words]
    max_score = max(scores)
    max_score_index = scores.index(max_score)
    # make sure the threshold is not larger than the max score, otherwise, no results will be returned

    # Function to calculate the score of a concatenated text
    def calculate_score(concatenated_text):
        concatenated_words = set(concatenated_text.split())
        return len(concatenated_words & answer_words) / len(answer_words)

        # Initialize variables

    current_indexes = [max_score_index]
    current_score = max_score
    no_improvement_count = 0

    # Concatenate indexes before the best match
    for i in range(max_score_index - 1, -1, -1):
        if scores[i] == 0:
            no_improvement_count += 1
            continue
        if no_improvement_count >= n:
            break
        new_indexes = [i] + current_indexes
        new_concatenated_text = " ".join(texts[index] for index in new_indexes)
        new_score = calculate_score(new_concatenated_text)
        if new_score > current_score:
            current_indexes = new_indexes
            current_score = new_score
            no_improvement_count = 0
        else:
            no_improvement_count += 1

    # Reset no_improvement_count for concatenating indexes after the best match
    no_improvement_count = 0

    # Concatenate indexes after the best match
    for i in range(max_score_index + 1, len(texts)):
        if scores[i] == 0:
            no_improvement_count += 1
            continue
        if no_improvement_count >= n:
            break
        new_indexes = current_indexes + [i]
        new_concatenated_text = " ".join(texts[index] for index in new_indexes)
        new_score = calculate_score(new_concatenated_text)
        if new_score > current_score:
            current_indexes = new_indexes
            current_score = new_score
            no_improvement_count = 0
        else:
            no_improvement_count += 1

    return current_indexes


def merge_metadatas(metadatas):
    if not metadatas:
        return {}
    metadata = metadatas[0]
    p1, p2, p3, p4 = metadata["coordinates"]["points"]
    corrdinate = [p1, p3]
    coordinates = [corrdinate]
    for metadata in metadatas[1:]:
        p1_, p2_, p3_, p4_ = metadata["coordinates"]["points"]
        if p2_[0] > p3[0]:
            corrdinate = [p1_, p3_]
            coordinates.append(corrdinate)
            p1, p2, p3, p4 = p1_, p2_, p3_, p4_
            continue
        p1 = (min(p1[0], p1_[0]), max(p1[1], p1_[1]))
        p2 = (min(p2[0], p2_[0]), max(p2[1], p2_[1]))
        p3 = (max(p3[0], p3_[0]), min(p3[1], p3_[1]))
        p4 = (max(p4[0], p4_[0]), min(p4[1], p4_[1]))
        corrdinate[0] = p1
        corrdinate[1] = p3

    page_number = metadata["page_number"]
    file_name = metadata["filename"]
    return {
        "file_name": file_name,
        "page_number": page_number,
        "coordinates": coordinates,
    }


def fetch_images(db, pdf_id, split_image_key):
    image_folder = os.environ.get("IMAGES_FOLDER", ".cache/images")
    image_folder = os.path.join(image_folder, str(pdf_id))
    if os.path.exists(image_folder) and os.listdir(image_folder):
        return

    os.makedirs(image_folder, exist_ok=True)
    table = db[split_image_key]
    select = table.filter(table["_source"].isin([pdf_id]))
    for doc in select.execute():
        image_path = doc[split_image_key].unpack()
        shutil.move(image_path, image_folder)


def get_related_merged_documents(
    db,
    contexts,
    chunk_key,
    split_image_key,
    match_text=None,
):
    """
    Convert contexts to a dataframe
    Will merge the same page
    """
    image_folder = os.environ.get("IMAGES_FOLDER", ".cache/images")
    source_ids = [source["_source"] for source in contexts]
    for source_id in source_ids:
        fetch_images(db, source_id, split_image_key)

    page_elements, page2score = groupby_source_elements(contexts, chunk_key)
    for (pdf_id, page_number), source_elements in page_elements.items():
        if match_text:
            match_indexes = rematch([e["text"] for e in source_elements], match_text)
            if not match_indexes:
                continue
            source_elements = [source_elements[i] for i in match_indexes]
        text = "\n".join([e["text"] for e in source_elements])
        metadata = merge_metadatas([e["metadata"] for e in source_elements])
        file_name = metadata["file_name"]
        coordinates = metadata["coordinates"]
        file_path = os.path.join(image_folder, str(pdf_id), f"{page_number-1}.jpg")
        if os.path.exists(file_path):
            img = draw_rectangle_and_display(file_path, coordinates)
        else:
            img = None
        score = round(page2score[(pdf_id, page_number)], 2)
        text = (
            f"**file_name**: {file_name}\n\n**score**: {score}\n\n**text:**\n\n{text}"
        )
        yield text, img


def groupby_source_elements(contexts, chunk_key):
    """
    Group pages for all contexts
    """
    from collections import defaultdict

    # Save the max score for each page
    page2score = {}
    page_elements = defaultdict(list)
    for source in contexts:
        outputs = source[chunk_key]
        source_elements = outputs["source_elements"]
        pdf_id = source["_source"]
        for element in source_elements:
            page_number = element["metadata"]["page_number"]
            page_elements[(pdf_id, page_number)].append(element)

        page_number = source_elements[0]["metadata"]["page_number"]
        score = source["score"]
        page2score[(pdf_id, page_number)] = max(page2score.get(page_number, 0), score)

    # Deduplicate elements in the page based on the num field
    for key, elements in page_elements.items():
        page_elements[key] = list({e["metadata"]["num"]: e for e in elements}.values())
        # Sort elements by num
        page_elements[key].sort(key=lambda e: e["metadata"]["num"])

    return page_elements, page2score


def draw_rectangle_and_display(image_path, relative_coordinates, expand=0.005):
    """
    Draw a rectangle on an image based on relative coordinates with the origin at the bottom-left
    and display it in Jupyter Notebook.

    :param image_path: Path to the original image.
    :param relative_coordinates: A list of (left, bottom, right, top) coordinates as a ratio of the image size.
    """
    from PIL import Image, ImageDraw

    with Image.open(image_path) as img:
        width, height = img.size
        draw = ImageDraw.Draw(img)

        # Convert relative coordinates to absolute pixel coordinates
        #
        for relative_coordinate in relative_coordinates:
            (left, top), (right, bottom) = relative_coordinate
            absolute_coordinates = (
                int(left * width),  # Left
                height - int(top * height),  # Top (inverted)
                int(right * width),  # Right
                height - int(bottom * height),  # Bottom (inverted)
            )

            if expand:
                absolute_coordinates = (
                    absolute_coordinates[0] - expand * width,
                    absolute_coordinates[1] - expand * height,
                    absolute_coordinates[2] + expand * width,
                    absolute_coordinates[3] + expand * height,
                )

            try:
                draw.rectangle(absolute_coordinates, outline="red", width=3)
            except Exception as e:
                logging.warn(
                    f"Failed to draw rectangle on image: {e}\nCoordinates: {absolute_coordinates}"
                )
        return img


class Processor(Model):
    chunk_key: str
    split_image_key: str

    def predict(self, contexts, match_text=None):
        return get_related_merged_documents(
            db=self.db,
            contexts=contexts,
            chunk_key=self.chunk_key,
            split_image_key=self.split_image_key,
            match_text=match_text,
        )


class Rag(Model):
    llm_model: Model
    prompt_template: str
    processor: Model
    vector_index: VectorIndex

    def __post_init__(self, *args, **kwargs):
        assert "{context}" in self.prompt_template, 'The prompt_template must include "{context}"'
        assert "{query}" in self.prompt_template, 'The prompt_template must include "{query}"'
        super().__post_init__(*args, **kwargs)

    def predict(self, query, top_k=5, format_result=False):
        vector_search_out = self.vector_search(query, top_k=top_k)
        key = self.vector_index.indexing_listener.key
        context = "\n\n---\n\n".join([x[key] for x in vector_search_out])
        
        prompt = self.prompt_template.format(context=context, query=query)
        output = self.llm_model.predict(prompt)
        result = {
            "answer": output,
            "docs": vector_search_out,
        }
        if format_result and self.processor:
            result["images"] = list(self.processor.predict(
                vector_search_out,
                match_text=output,
            ))
        return result

    def vector_search(self, query, top_k=5, format_result=False):
        logging.info(f"Vector search query: {query}")
        select = self.db[self.vector_index.indexing_listener.select.table].like(
            {self.vector_index.indexing_listener.key:query},
            vector_index=self.vector_index.identifier, 
            n=top_k,
        ).select()
        out = select.execute()
        if out:
            out = sorted(out, key=lambda x: x["score"], reverse=True)
        return out
