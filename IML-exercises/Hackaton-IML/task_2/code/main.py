from hackathon_code.parser import *



if __name__ == '__main__':
    train_label_0 = sys.argv[1]
    train_feature = sys.argv[2]
    train_label_1 = sys.argv[3]
    test = sys.argv[4]

    # proccess train
    r = get_uniq_labels()
    r = get_combinations(r)
    r = make_labels_dictionary(r)

    y0 = pandas.read_csv(train_label_0)
    y0 = preproccess_labels(y0)
    y0 = change_label(y0, r)

    r_inverse = labels_dictionary_inverse(r)

    y1 = pandas.read_csv(train_label_1).astype(float)
    X_train = pd.read_csv(train_feature)
    X_test = pd.read_csv(test)
    X_test = clean_unnecessary(X_test)
    X_train, y0, y1 = clean_unnecessary(X_train, True, y0, y1)


    # X_train1, X_test, y_train, y_test = train_test_split(X_train, y0, test_size=0.2, random_state=42)
    # task 0 :
    # task_0(X_train1, X_test, y_train,r_inverse, y_test)
    # X_train2, X_test, y_train, y_test = train_test_split(X_train, y1, test_size=0.2, random_state=42)
    # task_1(X_train2, X_test, y_train, y_test)
    task_0(X_train, X_test, y0, r_inverse)
    task_1(X_train, X_test, y1)

    # task 3
    # pca = PCA(
    #     n_components=2)  # Choose the number of components to visualize (2 for simplicity)
    # principal_components = pca.fit_transform(X_train)
    #
    # # Create a DataFrame to store the principal components
    # pc_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

    # sample_sizes = range(100, len(X_train), 100)
    # estimated_means = []
    # for n in sample_sizes:
    #     X_train, X_test, y_train, y_test  = train_test_split(X[:n], y0[:n], test_size=0.2, random_state=42)
    #     estimated_mean = task_1(X_train, X_test, y_train, y_test )
    #     estimated_means.append(estimated_mean)
    #
    # plt.scatter(sample_sizes, estimated_means)
    # plt.title('Mean Calculation by Sample Size')
    # plt.xlabel('Sample Number m')
    # plt.ylabel('Abs Distance from Expectation true value')
    # plt.show()
