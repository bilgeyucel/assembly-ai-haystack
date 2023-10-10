from typing import Dict, List, Any, Optional, Union
from pathlib import Path

import assemblyai as aai
from canals.serialization import default_to_dict, default_from_dict
from haystack.preview import component, Document

@component
class AssemblyAITranscriber:
    def __init__(self, api_key: str):
        aai.settings.api_key = api_key
        self.aai_transcriber = aai.Transcriber()
    
    def to_dict(self) -> Dict[str, Any]:
        return default_to_dict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AssemblyAITranscriber":
        return default_from_dict(cls, data)
    
    @component.output_types(documents=List[Document])    
    def run(
        self,
        file_url: Union[str, Path],
        iab_categories: Optional[bool] = None,
        speaker_labels: Optional[bool] = None,
    ):
        config = aai.TranscriptionConfig(speaker_labels=speaker_labels, iab_categories=iab_categories)
        transcript = self.aai_transcriber.transcribe(
            file_url,
            config=config
        )
        return {"documents": [Document(text=utterance.text, metadata={"speaker": utterance.speaker}) for utterance in transcript.utterances]}