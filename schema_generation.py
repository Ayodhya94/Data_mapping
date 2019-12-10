import random
import xml.etree.ElementTree as ET


def write_schema_xml_file(schema, number):
    # students = [1, 2]
    # assignments = [100, 101, 102]
    # scores = [0, 4, 10]
    # results = ET.Element("results")
    #
    # for s in students:
    #     for a in range(len(assignments)):
    #         result = ET.SubElement(results, "result")
    #         student = ET.SubElement(result, "student")
    #         assignment = ET.SubElement(result, "assignment")
    #         score = ET.SubElement(result, "score")
    #
    #         student.text = str(s)
    #         assignment.text = str(assignments[a])
    #         score.text = str(scores[a])
    #         results.append(result)
    #
    # tree = ET.ElementTree(results)
    # tree.write('test.xml')
    attribute_names = list(schema.keys())
    output_schema_attribute_names = []
    for attribute_name in attribute_names:
        attribute_name_variations = [attribute_name[:-1] + "1", "abc" + attribute_name[:-1],
                                     attribute_name.split("_")[0],
                                     attribute_name.split("_")[-1]]
        output_schema_attribute_names.append(
            attribute_name_variations[random.randint(0, len(attribute_name_variations) - 1)])

    number_of_instances = len(schema[attribute_names[0]])

    file_name = "Datasets/generated_schemas/schema_in_" + str(number + 1) + ".xml"
    root = ET.Element("root")
    for i in range(number_of_instances):
        instance_name = "instance_" + str(i)
        instance = ET.SubElement(root, instance_name)
        for attribute_name in attribute_names:
            attribute = ET.SubElement(instance, attribute_name)
            attribute.text = str(schema[attribute_name][i])
        # root.append(instance)
    tree = ET.ElementTree(root)
    tree.write(file_name)

    file_name = "Datasets/generated_schemas/schema_out_" + str(number + 1) + ".xml"

    root = ET.Element("root")
    for i in range(number_of_instances):
        instance_name = "instance_" + str(i)
        instance = ET.SubElement(root, instance_name)
        for j in range(len(output_schema_attribute_names)):
            attribute = ET.SubElement(instance, output_schema_attribute_names[j])
            attribute.text = str(schema[attribute_names[j]][i])
        # root.append(instance)
    tree = ET.ElementTree(root)
    tree.write(file_name)
    return


def populate_attributes(schema, attributes_of_schema):
    number_of_instances = 100
    for attribute in attributes_of_schema:
        attribute_name = attribute[0]
        attribute_data_type = attribute[1]
        path_to_data = "Lists/" + attribute_data_type[:-1] + ".txt"

        with open(path_to_data) as file:
            instances = file.readlines()

        schema[attribute_name] = []
        for i in range(number_of_instances):
            schema[attribute_name].append(instances[random.randint(0, len(instances)-1)])


def select_attribute_name(attribute_data_type):
    path_to_names = "Lists/names_" + attribute_data_type[:-1] + ".txt"
    with open(path_to_names) as file:
        attribute_names = file.readlines()
    return attribute_names[random.randint(0, len(attribute_names) - 1)]


def select_attribute_data_type(data_types):
    return data_types[random.randint(0, len(data_types) - 1)]


def get_available_data_types():
    path_to_data_types = "Lists/data_types.txt"
    with open(path_to_data_types) as file:
        data_types = file.readlines()
    return data_types


def create_attribute_list():
    attributes_list = []
    number_of_attributes = random.randint(1, 5)
    data_types = get_available_data_types()

    for i in range(number_of_attributes):
        attribute_data_type = select_attribute_data_type(data_types)
        attribute_name = select_attribute_name(attribute_data_type)
        attributes_list.append([attribute_name, attribute_data_type])

    return attributes_list


def generate_schemas(number_of_schema):
    schema = {}
    for i in range(number_of_schema):
        attributes_of_schema = create_attribute_list()
        print("________________")
        print("SCHEMA " + str(i))
        print("________________")
        for attribute in attributes_of_schema:
            print(attribute[0][:-1])
        print("_________________")
        populate_attributes(schema, attributes_of_schema)
        write_schema_xml_file(schema, i)


def main():
    number_of_schema = 10
    generate_schemas(number_of_schema)


if __name__ == "__main__":
    main()


""" XML """
"""
import xml.etree.cElementTree as ET

students = [1,2]
assignments=[100,101,102]
scores=[0,4,10]
results = ET.Element("results")

for s in students:    
    for a in range(len(assignments)):
        result = ET.SubElement(results,"result")
        student = ET.SubElement(result,"student")
        assignment = ET.SubElement(result,"assignment")
        score = ET.SubElement(result,"score")

        student.text = str(s)
        assignment.text = str(assignments[a])
        score.text = str(scores[a])
        results.append(result)

tree = ET.ElementTree(results)
tree.write('test.xml')
"""