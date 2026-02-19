from __future__ import annotations
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

class AnalysisResult(BaseModel):
  body_type: str = Field(description="Predicted body type category")
  face_type: str = Field(description="Predicted face type/shape")
  skin_tone: str = Field(description="Predicted skin tone category")
  outfits: List[str] = Field(default_factory=list, description="Detected outfit categories")
  debug: Optional[Dict[str, float]] = Field(default=None, description="Optional debug metrics")