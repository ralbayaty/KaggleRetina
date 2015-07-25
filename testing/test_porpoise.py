from vowpal_porpoise import VW

# Initialize the model
vw = VW(moniker='test',    # a name for the model
        passes=10,         # vw arg: passes
        loss='quadratic',  # vw arg: loss
        learning_rate=10,  # vw arg: learning_rate
        l1=0.01)           # vw arg: l1

# Inside the with training() block a vw process will be 
# open to communication
with vw.training():
    for instance in ['1 |big red square',\
                      '0 |small blue circle']:
        vw.push_instance(instance)

    # here stdin will close
# here the vw process will have finished

# Inside the with predicting() block we can stream instances and 
# acquire their labels
with vw.predicting():
    for instance in ['1 |large burnt sienna rhombus',\
                      '0 |little teal oval']:
        vw.push_instance(instance)

# Read the predictions like this:
predictions = list(vw.read_predictions_())