import schema_generation
import schema_matching_general
import statistics


def write_test_file_for_data_mapper(input_schema, output_schema):
    with open("Datasets/courses_schemas_test.txt", 'w') as file:
        file.write("Datasets/generated_schemas/" + input_schema)
        file.write("Datasets/generated_schemas/" + output_schema)


def get_mappings_from_model(input_schema_file_path, output_schema_file_path):
    write_test_file_for_data_mapper(input_schema_file_path, output_schema_file_path)
    mapping_list = schema_matching_general.main()
    return mapping_list


def get_correct_mappings(mapping_file_path):
    with open("Datasets/generated_schemas/" + mapping_file_path[:-1]) as file:
        mapping_list = file.readlines()

    number_of_mappings = int(len(mapping_list)/2)
    mappings = {}
    for i in range(number_of_mappings):
        mappings[mapping_list[2*i][:-1]] = mapping_list[2*i + 1][:-1]

    return mappings


def calculate_accuracy(mappings_of_model, correct_mappings):
    number_of_correct_mappings = 0
    number_of_incorrect_mappings = 0
    total_number_of_attributes = len(correct_mappings)
    for map in mappings_of_model:
        try:
            if correct_mappings[map[0]] == map[1]:
                number_of_correct_mappings += 1
            else:
                number_of_incorrect_mappings += 1
        except (IndexError):
            continue
    precision = number_of_correct_mappings/(number_of_correct_mappings+number_of_incorrect_mappings) * 100
    recall = number_of_correct_mappings/ total_number_of_attributes * 100
    F1_score = 2 * precision * recall / (precision + recall)

    return [precision, recall, F1_score]


def main():
    schema_sets_to_test = "Datasets/test_data.txt"
    with open(schema_sets_to_test) as file:
        file_paths_to_test_schemas = file.readlines()

    number_of_test_sets = int(len(file_paths_to_test_schemas)/3)
    precision = []
    recall = []
    F1_score = []
    for i in range(number_of_test_sets):
        mappings_of_model = get_mappings_from_model(file_paths_to_test_schemas[3*i], file_paths_to_test_schemas[3*i + 1])
        correct_mappings = get_correct_mappings(file_paths_to_test_schemas[3*i + 2])
        accuracy = calculate_accuracy(mappings_of_model, correct_mappings)

        precision.append(accuracy[0])
        recall.append(accuracy[1])
        F1_score.append(accuracy[2])

        print(str(i))
        print(mappings_of_model)
        print(correct_mappings)
        print(accuracy)

    print("Precision: %f" % statistics.mean(precision))
    print("Recall: %f" % statistics.mean(recall))
    print("F1 score: %f" % statistics.mean(F1_score))


if __name__ == "__main__":
    main()

