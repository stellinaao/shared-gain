from sg.models import Encoder


class TestEncoder:
    def setup_method(self):
        # arrange
        subj_id = "MR82"
        sess_id = "20251027_152036"

        self.encoder = Encoder(
            subj_id=subj_id,
            sess_id=sess_id,
        )

    def test_consistent_num_trials(self):
        # act
        self.encoder.get_data()
        self.encoder.build_dm()

        # assert
        assert self.encoder.num_trials == self.encoder.trial_data.shape[0]


class TestStrategyEncoder:
    def setup_method(self):
        # arrange
        pass

    def test_something(self):
        # act
        pass

        # assert
        pass


class TestShuffledEncoder:
    def setup_method(self):
        # arrange
        pass

    def test_something(self):
        # act
        pass

        # assert
        pass
