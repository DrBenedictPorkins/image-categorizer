# Sample images

Eleven small images for checking an installation end to end:

```bash
uv run python main.py samples --provider ollama
```

They cover the kinds of photos the tool is built to separate: people, pets, food,
places, documents, drawings, and accidental shots. Results and the report are
written into this folder and are ignored by git.

| File | Source | Author | License |
|---|---|---|---|
| `dog.jpg` | [Golden retriever alert (30190754125).jpg](https://commons.wikimedia.org/wiki/File:Golden_retriever_alert_(30190754125).jpg) | David Whelan | CC0 |
| `cat.jpg` | [Young tabby cat keeping watch.jpg](https://commons.wikimedia.org/wiki/File:Young_tabby_cat_keeping_watch.jpg) | W.carter | CC0 |
| `astronaut-with-dogs.jpg` | [NASA astronaut Leland D. Melvin with his dogs Jake and Scout.jpg](https://commons.wikimedia.org/wiki/File:NASA_astronaut_Leland_D._Melvin_with_his_dogs_Jake_and_Scout.jpg) | Robert Markowitz | Public domain |
| `pancakes.jpg` | [Eating Pancakes (Unsplash).jpg](https://commons.wikimedia.org/wiki/File:Eating_Pancakes_(Unsplash).jpg) | Gabriel Gurrola gabrielgurrola | CC0 |
| `birthday-cake.jpg` | [Tim Tams and Nuts on a Birthday Cake With Candles.jpg](https://commons.wikimedia.org/wiki/File:Tim_Tams_and_Nuts_on_a_Birthday_Cake_With_Candles.jpg) | Brassluff | CC0 |
| `lake.jpg` | [Moraine Lake 17092005.jpg](https://commons.wikimedia.org/wiki/File:Moraine_Lake_17092005.jpg) | Gorgo | Public domain |
| `building.jpg` | [Parque Avenida Building in Paulista Avenue.jpg](https://commons.wikimedia.org/wiki/File:Parque_Avenida_Building_in_Paulista_Avenue.jpg) | Wilfredor | CC0 |
| `handwritten-letter.jpg` | [Handwritten letter of Nana Phadnavis.jpg](https://commons.wikimedia.org/wiki/File:Handwritten_letter_of_Nana_Phadnavis.jpg) | Unknown author | Public domain |
| `finger-drawing.png` | Generated for this repository | - | CC0 |
| `accidental-motion-blur.jpg` | `dog.jpg`, rotated and motion-blurred | David Whelan | CC0 |
| `accidental-pocket.jpg` | Crop of `lake.jpg`, darkened, with added noise | Gorgo | Public domain |

Photos are from Wikimedia Commons, resized to at most 1024 pixels on the long side.
