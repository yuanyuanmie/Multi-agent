#-------------------------------------------
# Author: Alkaid
# Date: March 6 2025
# Last Revision: April 29 2025
# Core functions for Report generations
# Providing 'Planning--Researching--Writing--Reviewing--Revising' pipeline																				   	  
# filename workflow.py																																																			#------------------------------------------- 
import os, sys, re, asyncio, json, copy
import numpy as np
import time
import pandas as pd
from pathlib import Path

from dataclasses import dataclass
from typing import Tuple, Optional
from typing_extensions import TypedDict
from IPython.display import Image, display
from docx import Document

from datetime import datetime, date
current_dateTime = datetime.now()
current_date = date.today()

root_dir = '/home/user/users/llmporject'
sys.path.append(root_dir)
#import orm
from xengine import Text, latex
from xengine.Text import Chapter
from research import Research, ReportMeta
from agents import ReportState, action_parser, action_master, action_researcher, action_writer, action_reviewer, action_revisor, check_format
from agents import select_revise_or_end, select_write_or_end, select_validate_or_end

from logger import LocalLogger

from langchain_community.callbacks.openai_info import OpenAICallbackHandler
from langgraph.graph import END, START, StateGraph

#------update_and_backup----------------------------------------------------------
@dataclass
class TaskConfig:
    title:           str
    constrain:       str
    update_db:      bool
    task_id:         int
    upload_path:     str
    style: Optional[str] =       None
    meta: Optional[ReportMeta] = None
#-----StateMachine-------------------------------------------------------------------
class StateMachine():
    def __init__(self):
        self.graph = self._build_graph()

    def _build_graph(self):
        # Initialize graph nodes
        graph = StateGraph(ReportState)
        graph.add_node("parser", action_parser)
        graph.add_node("master", action_master)
        graph.add_node("researcher", action_researcher)
        graph.add_node("writer", action_writer)
        graph.add_node("reviewer", action_reviewer)
        graph.add_node("revisor", action_revisor)
        # Initialize parameters
        graph.add_edge(START, "parser")
        graph.add_edge("parser", "master")
        graph.add_edge("master", "researcher")
        graph.add_edge("researcher", "writer")
        graph.add_edge("writer", "reviewer")
        graph.add_conditional_edges("reviewer", select_revise_or_end)
        graph.add_conditional_edges("revisor", select_write_or_end)
        return graph.compile()

    def show_graph(self):
        self.image = Image(self.graph.get_graph().draw_mermaid_png())
        display(self.image)

    async def ainvoke(self,state,callback_handler):
        self.state = await self.graph.ainvoke(state,
                                              config={"recursion_limit": 50,
                                                      "callbacks": [callback_handler],})
        return self.state

#------Task Runner---------------------------------------------------
class ReportWorkflowRunner():
    def __init__(self, config: TaskConfig, logger: LocalLogger):
        self.state_machine = StateMachine()
        self.logger = logger
        self._unpack_config(config)

    def _unpack_config(self, config: TaskConfig):
        self.config = config
        self.title = config.title
        self.update_db = config.update_db
        self.task_id = config.task_id
        self.constrain = config.constrain
        self.style = config.style
        self.upload_path = config.upload_path
        self.mode = 'market_frontpage' if config.style == '市场分析' else 'docx'
        self.callback_handler = OpenAICallbackHandler()
        meta = ReportMeta(
            style=config.style,
            research_object=None,
            sector=None,
            file_path=config.upload_path
        )
        self.state = {
            "title": self.title, 
            "constrain": self.constrain,
            "skip_review": True,
            "update_db": self.update_db,
            "current_chapter": 1,
            "task_id": self.task_id,
            "meta": meta,
            "logger": self.logger
        }
        

    def read_state(self, state: ReportState):
        self.state = state

    async def run(self):
        self.logger.general_message("Report generating.")
        self.state = await self.state_machine.ainvoke(state = self.state,callback_handler = self.callback_handler)
        self.logger.general_message("Report generated.")
        return self.state

#--------Saver---------------------------------------------------------------
class WorkflowSaver:
    def __init__(self, state, title, mode, user_id, logger):
        self.state = state
        self.title = title
        self.mode = mode
        self.user_id = str(user_id)
        self.current_datetime = datetime.now()
        self.user_path = Path(self.user_id)
        self._ensure_user_dir()
        self.logger = logger

    def _ensure_user_dir(self):
        if not self.user_path.exists():
            self.user_path.mkdir(parents=True)
            self.logger.general_message("User path created.")

    def _generate_filename(self, suffix: str) -> str:
        timestamp = self.current_datetime.strftime("%Y%m%d_%H%M%S")
        return f"{self.title}_{timestamp}.{suffix}"

    def receive_state(self, state: ReportState):
        self.state = state

    def receive_and_save(self, state: ReportState) -> Tuple[str, str]:
        self.receive_state(state)
        report_path = self._save_report()
        json_path = self._save_state()
        return report_path, json_path

    def _save_report(self, version='validated_version') -> str:
        if self.mode == 'docx':
            path = self._save_docx(version)
        elif self.mode in ['industry_frontpage', 'market_frontpage']:
            path = self._save_pdf()
        else:
            raise ValueError(f"Unsupported report mode: {self.mode}")
        self.logger.general_message(f"Report saved to {path}")
        return str(path)

    def _save_docx(self, version: str) -> Path:
        document = Document()
        Text.add_title(document, self.title, style='report')
        heading = document.add_heading()

        total_chapter = self.state['total_chapter']
        for i in range(total_chapter):
            chapter = self.state[f"chapter_{i+1}"]
            if chapter['validate']:
                Text.add_chapter_with_reference(document,
                                                chapter_title=chapter['chapter_title'],
                                                style='report',
                                                paragraph_reference=chapter['para_ref'],
                                                local_db=chapter['local_db'])
            else:
                Text.add_chapter(document,
                                 chapter_title=chapter['chapter_title'],
                                 content=chapter[version],
                                 style='report',
                                 reference=chapter['reference'])

        filename = self._generate_filename('docx')
        path = self.user_path / filename
        document.save(path)
        return path

    def _save_pdf(self) -> Path:
        text = {
            'title': self.title,
            'text_1': self.state["chapter_1"]['validated_version'],
            'text_2': self.state["chapter_2"]['validated_version'],
            'text_3': self.state["chapter_3"]['validated_version'],
            'logo': 'logo.png',
            'chart': 'chart.png',
            'date': self.current_datetime.strftime('%Y-%m-%d'),
        }
        filename = self._generate_filename('pdf')
        path = self.user_path / filename
        latex.latex_to_pdf(text=text, file_title=path.stem)
        return path

    def _save_state(self) -> str:
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)

        filename = self._generate_filename('json')
        path = self.user_path / filename
        state_to_save = copy.deepcopy(self.state)
        state_to_save.pop('logger', None)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(state_to_save, f, ensure_ascii=False, indent=4, cls=NpEncoder)
        self.logger.general_message(f"数据已成功保存到 '{path}'")
        return str(path)