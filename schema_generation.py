import random
import string
import re
import xml.etree.ElementTree as ET


def write_xml_file(number_of_instances, attribute_names, schema, file_name):
    root = ET.Element("root")
    for i in range(number_of_instances):
        instance_name = "instance_" + str(i)
        instance = ET.SubElement(root, instance_name)
        for attribute_name in attribute_names:
            attribute = ET.SubElement(instance, attribute_name)
            attribute.text = str(schema[attribute_name][i])
    tree = ET.ElementTree(root)
    tree.write(file_name)


def save_schema_set(input_schema, output_schema, number, number_of_instances):
    input_schema_attribute_names = list(input_schema.keys())
    output_schema_attribute_names = list(output_schema.keys())

    input_file_name = "Datasets/generated_schemas/schema_in_" + str(number + 1) + ".xml"
    write_xml_file(number_of_instances, input_schema_attribute_names, input_schema, input_file_name)

    output_file_name = "Datasets/generated_schemas/schema_out_" + str(number + 1) + ".xml"
    write_xml_file(number_of_instances, output_schema_attribute_names, output_schema, output_file_name)
    # root = ET.Element("root")
    # for i in range(number_of_instances):
    #     instance_name = "instance_" + str(i)
    #     instance = ET.SubElement(root, instance_name)
    #     for attribute_name in input_schema_attribute_names:
    #         attribute = ET.SubElement(instance, attribute_name)
    #         attribute.text = str(schema[attribute_name][i])
    # tree = ET.ElementTree(root)
    # tree.write(file_name)

    # file_name = "Datasets/generated_schemas/schema_out_" + str(number + 1) + ".xml"
    # root = ET.Element("root")
    # for i in range(number_of_instances):
    #     instance_name = "instance_" + str(i)
    #     instance = ET.SubElement(root, instance_name)
    #     for j in range(len(output_schema_attribute_names)):
    #         attribute = ET.SubElement(instance, output_schema_attribute_names[j])
    #         attribute.text = str(schema[input_schema_attribute_names[j]][i])
    # tree = ET.ElementTree(root)
    # tree.write(file_name)
    return


def populate_attributes(schema, attributes_of_schema, number_of_instances):
    for attribute in attributes_of_schema:
        attribute_name = attribute[0]
        attribute_data_type = attribute[1]
        path_to_data = "Lists/" + attribute_data_type[:-1] + ".txt"

        with open(path_to_data) as file:
            instances = file.readlines()

        schema[attribute_name] = []
        for i in range(number_of_instances):
            schema[attribute_name].append(instances[random.randint(0, len(instances)-1)][:-1])


def create_random_string():
    random_string = ""
    letter_count = random.randint(0, 4)
    for i in range(letter_count):
        random_string = random_string + random.choice(string.ascii_letters)
    return random_string + "_"


def variate_attribute_names(attribute_names):
    variated_list = []
    for attribute_name in attribute_names:
        name = attribute_name[0]
        data_type = attribute_name[1]

        split_name = list(
            filter(''.__ne__, (re.split('[_ :]', re.sub(r"([A-Z]+[a-z0-9_\W])", r" \1", name + "_").lower()))))

        attribute_name_variations = [name + str(random.randint(1, 100)), create_random_string() + name, split_name[0],
                                     split_name[-1], name.lower(), "_".join(split_name)]

        variated_list.append(
            [attribute_name_variations[random.randint(0, len(attribute_name_variations) - 1)], data_type])
    return variated_list


def select_attribute_name(attribute_data_type):
    path_to_names = "Lists/names_" + attribute_data_type[:-1] + ".txt"
    with open(path_to_names) as file:
        attribute_names = file.readlines()
    return attribute_names[random.randint(0, len(attribute_names) - 1)][:-1]


def select_attribute_data_type(data_types):
    return data_types[random.randint(0, len(data_types) - 1)]


def get_available_data_types():
    path_to_data_types = "Lists/data_types.txt"
    with open(path_to_data_types) as file:
        data_types = file.readlines()
    return data_types


def create_attribute_list():
    attributes_list = []
    number_of_attributes = random.randint(1, 10)
    data_types = get_available_data_types()

    for i in range(number_of_attributes):
        attribute_data_type = select_attribute_data_type(data_types)
        attribute_name = select_attribute_name(attribute_data_type)
        attributes_list.append([attribute_name, attribute_data_type])
    return attributes_list


def generate_schemas(number_of_schemas):
    for i in range(number_of_schemas):

        # Generate attributes
        input_schema = {}
        output_schema = {}
        attributes_of_input_schema = create_attribute_list()
        attributes_of_output_schema = variate_attribute_names(attributes_of_input_schema)
        print("________________")
        print("SCHEMA " + str(i))
        print("________________")
        for attribute in attributes_of_input_schema:
            print(attribute[0])
        print("_________________")
        for attribute in attributes_of_output_schema:
            print(attribute[0])
        print("_________________")

        # Populate schemas
        number_of_instances = 100
        populate_attributes(input_schema, attributes_of_input_schema, number_of_instances)
        populate_attributes(output_schema, attributes_of_output_schema, number_of_instances)

        # Save schema sets
        save_schema_set(input_schema, output_schema, i, number_of_instances)

        # Write correct mappings to files
        answer_file_name = "Datasets/generated_schemas/mappings_" + str(i) + ".txt"
        with open(answer_file_name, 'w+') as f:
            for j in range(len(attributes_of_input_schema)):
                f.write(attributes_of_input_schema[j][0] + "\n")
                f.write(attributes_of_output_schema[j][0] + "\n")


def main():
    number_of_schemas = 10
    generate_schemas(number_of_schemas)


if __name__ == "__main__":
    main()
