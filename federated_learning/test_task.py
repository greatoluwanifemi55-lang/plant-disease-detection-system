from plant_fl.task import load_model, load_data

dataset, classes = load_data("Oyo")

print("Number of classes:", len(classes))
print(classes)

model = load_model()

model.summary()