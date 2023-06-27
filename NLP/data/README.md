# Data

* Document_context_data

  * Raw_data: contain 28 document about 28 diseases
  * Chunking: contain 1005 context chunking from raw_data
* Conversation_data(Lương)

  * conversation_data_all: contain 28 json file corresponding with 28 diseases
  * conversation_data_dialogpt: contain 7303 total_data with 5842 train, 729 validation and 730 test samples
  * ```
    {
      "context 0": "",
      "context 1": "",
      "context 2": "",
      "context 3": "",
      "context 4": "",
      "response": ""
    }
    ```
  * conversation_data_godel: contain 7303 total_data with 5842 train, 729 validation and 730 test samples
    ```
    {
      "dialog":[],
      "knowledge": "",
      "response": ""
    }
    ```
* Dialog_knowledge_data

  * dialog_knowledge_data_all: contain 1005 file corresponding with 1005 contexts + 27 invalid.txt file that contain error sample while generating data
  * dialog_knowledge_data_dialogpt: contain 28535 total_data with 22821 train, 2854 validation and 2851 test samples
  * ```
    {
      "context 0": "",
      "context 1": "",
      "context 2": "",
      "context 3": "",
      "context 4": "",
      "context 5": "",
      "context 6": "",
      "context 7": "",
      "context 8": "",
      "context 9": "",
      "response": ""
    }

    ```
  * dialog_knowledge_data_godel: contain 28535 total_data with 22821 train, 2854 validation and 2851 test samples
    ```
    {
      "dialog":[],
      "knowledge": "",
      "response": ""
    }

    ```
