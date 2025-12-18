import json

def check_polygon_coordinate_types(json_path, max_examples=5):
    with open(json_path, "r") as f:
        coco = json.load(f)

    total_coords = 0
    int_coords = 0
    float_coords = 0
    other_coords = 0

    examples = []

    for ann in coco.get("annotations", []):
        segmentation = ann.get("segmentation", [])

        # pomijamy RLE (jeśli występuje)
        if not isinstance(segmentation, list):
            continue

        for polygon in segmentation:
            for coord in polygon:
                total_coords += 1

                if isinstance(coord, int):
                    int_coords += 1
                elif isinstance(coord, float):
                    float_coords += 1
                    if len(examples) < max_examples:
                        examples.append(coord)
                else:
                    other_coords += 1
                    if len(examples) < max_examples:
                        examples.append(coord)

    print("=== RAPORT WSPÓŁRZĘDNYCH POLYGONÓW ===")
    print(f"Łącznie współrzędnych: {total_coords}")
    print(f"int:   {int_coords}")
    print(f"float: {float_coords}")
    print(f"inny typ: {other_coords}")

    if examples:
        print("\\nPrzykładowe problematyczne wartości:")
        for e in examples:
            print(f"  {e} (type={type(e)})")


# ====== użycie ======
check_polygon_coordinate_types("track2/buildings_only_train.json")
