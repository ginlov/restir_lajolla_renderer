import xml.etree.ElementTree as ET

def read_xml(file_path):
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        return tree, root
    except ET.ParseError as e:
        print(f"Error parsing XML file: {e}")
        return None, None

def transform_elements(root, attributes_to_transform):
    if root is not None:
        for element in root.iter():
            if element.tag == 'rgb' and element.attrib.get('name') in attributes_to_transform:
                value = element.attrib.get('value')
                if value:
                    # Extract the first float element from the value
                    first_float = value.split()[0]
                    # Create a new float element with the same attributes except for the value
                    new_element = ET.Element('float', attrib={**element.attrib, 'value': first_float})
                    # Replace the old element with the new one
                    parent = tree.find(".//" + element.tag + "[@name='" + element.attrib['name'] + "']/..")
                    if parent is not None:
                        parent.remove(element)
                        parent.append(new_element)

def print_xml_elements(root):
    if root is not None:
        for child in root:
            print(f"Tag: {child.tag}, Attributes: {child.attrib}")
            print_xml_elements(child)

if __name__ == "__main__":
    file_path = "scenes/car2/car.xml"
    new_file = "scenes/car2/car2.xml"
    attributes_to_transform = [
        "roughness", "subsurface", "anisotropic", "eta", "clearcoatGloss", "sheenTint", "sheen_tint",
        "specularTransmission", "specular_transmission", "specTrans", "spec_trans", "metallic", "specular",
        "sheen", "clearcoat", "spec_tint"
    ]
    tree, root = read_xml(file_path)
    if root:
        transform_elements(root, attributes_to_transform)
        tree.write(new_file)  # Save the modified XML back to the file
        print_xml_elements(root)