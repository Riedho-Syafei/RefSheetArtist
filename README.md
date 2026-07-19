# RefSheetArtist
A character reference sheet generator that utilizes FLUX.2 Klein 4B as the backend.

## Make sure to put your Hugging Face token to download_flux.py and run it first
```
python download_flux.py
```

## Usage

### Fresh generation
```
python refsheet_artist.py --name "My character" --prompt "My character description..."
```
```
python refsheet_artist.py --name "Orange Tabby Blacksmith" --prompt "a stocky orange tabby cat-person blacksmith, exactly one scar, located only on the upper left arm, no other scars or wounds anywhere on the body, empty paws, holding nothing, no weapons, no tools, wears a leather apron, semi-realistic style"
```

### Fix just one bad view against an existing saved project
```
python refsheet_artist.py --name "My character" --regenerate-view side
```
```
python refsheet_artist.py --name "Orange Tabby Blacksmith" --regenerate-view side
```

### Just rebuild the sheet image (e.g. after tweaking compositor.py's cell size) — no generation at all
```
python refsheet_artist.py --name "My character" --recomposite-only
```
```
python refsheet_artist.py --name "Orange Tabby Blacksmith" --recomposite-only
```
