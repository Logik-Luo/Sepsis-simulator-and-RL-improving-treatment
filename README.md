# Sepsis-simulator-and-RL-improving-treatment
create 100 fake patients' records (you could define the amount you want) at once, however the acc of outcome model (which directly affects the realism of generated labels) is not too high,
so you would witness negative rewards on test set (generated states and labels, labels were used to define rewards) when RL model trained on real dataset.
