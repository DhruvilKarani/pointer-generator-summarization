import torchtext


class SummarizationDataset:

    def __init__(self, path_to_csv, text_column, summary_column, SOS="<SOS>", EOS="<EOS>", PAD="<PAD>", test_ratio=0.2):
        self.csv_path = path_to_csv
        self.text_column = text_column
        self.summary_column = summary_column
        self.text_field = None
        self.summary_field = None
        self.sos = SOS
        self.eos = EOS
        self.pad = PAD
        self.train_ratio = 1-test_ratio

    def _build_fields(self):
        text_field = torchtext.data.Field(
            sequential = True,
            init_token = self.sos,
            eos_token = self.eos,
            tokenize="spacy",
            include_lengths=True,
            pad_token=self.pad,
            batch_first=True
        )

        summary_field = torchtext.data.Field(
            sequential = True,
            init_token = self.sos,
            eos_token = self.eos,
            tokenize="spacy",
            include_lengths=True,
            pad_token=self.pad,
            batch_first=True,
            is_target=True
        )

        self.text_field = text_field
        self.summary_field = summary_field

    def get_datasets(self):
        if self.text_field == None or self.summary_field == None:
            self._build_fields()
        
        fields = {
            self.text_column: (self.text_column, self.text_field), 
            self.summary_column: (self.summary_column, self.summary_field)
            }

        dataset = torchtext.data.TabularDataset(self.csv_path, format='csv', fields=fields, skip_header=False)
        return dataset.split(split_ratio=self.train_ratio)


if __name__ == '__main__':
    CSV_PATH = "../../data/processed/data.csv"
    summ_data = SummarizationDataset(CSV_PATH, "text", "summary")
    train_dataset, text_dataset = summ_data.get_datasets()