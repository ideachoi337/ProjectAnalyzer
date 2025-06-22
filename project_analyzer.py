import sys
import os
import ast
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QTreeWidget, QTreeWidgetItem,
    QVBoxLayout, QWidget, QPushButton, QPlainTextEdit, QTextEdit,
    QHBoxLayout, QSplitter, QProgressBar
)
from PyQt5.QtCore import Qt, QRunnable, QThreadPool, pyqtSlot, pyqtSignal, QObject
import requests
import markdown
from pygments.formatters import HtmlFormatter
import weakref
from collections import deque

from PyQt5.QtGui import QSyntaxHighlighter, QTextCharFormat, QColor, QFont
import re
import markdown
count = 0

class PythonHighlighter(QSyntaxHighlighter):
    def __init__(self, document):
        super().__init__(document)

        def format(color, style=''):
            _format = QTextCharFormat()
            _format.setForeground(QColor(color))
            if 'bold' in style:
                _format.setFontWeight(QFont.Bold)
            if 'italic' in style:
                _format.setFontItalic(True)
            return _format

        self.highlighting_rules = []

        keyword_format = format('blue', 'bold')
        keywords = [
            'and', 'as', 'assert', 'break', 'class', 'continue', 'def',
            'del', 'elif', 'else', 'except', 'False', 'finally', 'for',
            'from', 'global', 'if', 'import', 'in', 'is', 'lambda', 'None',
            'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'True',
            'try', 'while', 'with', 'yield'
        ]
        for kw in keywords:
            pattern = r'\b' + kw + r'\b'
            self.highlighting_rules.append((re.compile(pattern), keyword_format))

        string_format = format('darkgreen')
        self.highlighting_rules.append((re.compile(r'"[^"\\]*(\\.[^"\\]*)*"'), string_format))
        self.highlighting_rules.append((re.compile(r"'[^'\\]*(\\.[^'\\]*)*'"), string_format))

        comment_format = format('gray', 'italic')
        self.highlighting_rules.append((re.compile(r'#.*'), comment_format))

        number_format = format('purple')
        self.highlighting_rules.append((re.compile(r'\b[0-9]+(\.[0-9]*)?\b'), number_format))

    def highlightBlock(self, text):
        for pattern, fmt in self.highlighting_rules:
            for match in pattern.finditer(text):
                start, end = match.span()
                self.setFormat(start, end - start, fmt)


class WorkerSignals(QObject):
    finished = pyqtSignal(str)

class ChatWorker(QRunnable):
    def __init__(self, request_fn, inst, summary, type, callback):
        super().__init__()
        self.request_fn = request_fn
        self.inst = inst
        self.signals = WorkerSignals()
        self.summary = summary
        self.type = type
        self.signals.finished.connect(callback)

    @pyqtSlot()
    def run(self):
        try:
            response = self.request_fn(self.type, self.inst, self.summary)
            desc = response['message']
        except:
            desc = ''
        self.signals.finished.emit(desc)

class LLMWorker(QRunnable):
    def __init__(self, request_fn, snippet, cache, type, callback):
        super().__init__()
        self.request_fn = request_fn
        self.snippet = snippet
        self.signals = WorkerSignals()
        self.cache = cache
        self.type = type
        self.signals.finished.connect(callback)

    @pyqtSlot()
    def run(self):
        try:
            response = self.request_fn(self.type, self.snippet)
            desc = response['message']
        except:
            desc = ''
        self.signals.finished.emit(desc)
        if self.cache is not None:
            self.cache['desc'] = desc

class CodeRequests:
    def __init__(self, addr='http://127.0.0.1:8002'):
        self.addr = addr

    def request(self, type, data):
        response = requests.post(f"{self.addr}/{type}", json={"code": data})
        return response.json()

class ChatRequests:
    def __init__(self, addr='http://127.0.0.1:8002'):
        self.addr = addr

    def request(self, type, inst, summary):
        response = requests.post(f"{self.addr}/{type}", json={"instruction": inst, "summary": summary})
        return response.json()

class FileBrowser(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Python Project Analyzer")
        self.setGeometry(100, 100, 1200, 700)

        self.project_path = None
        self.file_cache = {}
        self.current_file = None
        self.code_request = CodeRequests()
        self.chat_request = ChatRequests()
        self.last_clicked = None
        self.task_queue = deque()
        self.total_tasks = 0
        self.completed_tasks = 0

        open_btn = QPushButton("Open Project")
        open_btn.clicked.connect(self._open_folder)

        self.project_tree = QTreeWidget()
        self.project_tree.setHeaderLabels(["Project Tree"])
        self.project_tree.itemClicked.connect(self._on_project_item_clicked)

        self.code_viewer = QPlainTextEdit(readOnly=True)
        self.highlighter = PythonHighlighter(self.code_viewer.document())
        self.summary_viewer = QTextEdit(readOnly=True)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)

        summary_panel = QWidget()
        summary_layout = QVBoxLayout()
        summary_layout.setContentsMargins(0, 0, 0, 0)
        summary_layout.setSpacing(2)
        summary_layout.addWidget(self.summary_viewer)
        summary_layout.addWidget(self.progress_bar)
        summary_panel.setLayout(summary_layout)

        self.structure_tree = QTreeWidget()
        self.structure_tree.setHeaderLabels(["Structure (Class / Function)"])
        self.structure_tree.itemClicked.connect(self._on_structure_item_clicked)

        self.chat_history_html = """
        <style>
        .chat-box {
            font-family: sans-serif;
            font-size: 14px;
            line-height: 1.6;
            padding: 5px;
        }

        .user {
            display: flex;
            justify-content: flex-end;
            margin-bottom: 10px;
        }

        .user-msg {
            background-color: #e3f2fd;
            padding: 10px;
            border-radius: 10px;
            max-width: 80%;
            word-wrap: break-word;
            white-space: pre-wrap;
            box-sizing: border-box;
        }

        .user-msg ul, .user-msg ol {
            padding-left: 20px;
            margin: 0;
        }

        .user-msg li {
            margin-bottom: 4px;
        }

        .user-msg pre {
            background-color: #f1f1f1;
            padding: 8px;
            border-radius: 6px;
            overflow-x: auto;
            margin: 6px 0;
        }
        </style>
        <div class="chat-box">
        """
        self.chat_placeholder_default = "Enter your message..."
        self.chat_placeholder_waiting = "Code analysis is still in progress. Please wait until all files are processed."

        self.chat_input = QPlainTextEdit()
        self.chat_input.setMinimumHeight(60)
        self.chat_input.setEnabled(True)
        self.chat_input.setPlaceholderText(self.chat_placeholder_default)

        self.chat_output = QTextEdit(readOnly=True)
        self.chat_output.setPlaceholderText("Chat response will appear here.")

        send_btn = QPushButton("Send")
        send_btn.setFixedHeight(40)
        send_btn.clicked.connect(self._send_chat_message)
        send_btn.setEnabled(True)
        self.send_btn = send_btn 

        chat_layout = QVBoxLayout()
        chat_layout.addWidget(self.chat_output, stretch=5)
        chat_layout.addWidget(self.chat_input, stretch=1)
        chat_layout.addWidget(send_btn)

        chat_widget = QWidget()
        chat_widget.setLayout(chat_layout)

        project_tree_with_button = QWidget()
        project_layout = QVBoxLayout()
        project_layout.setContentsMargins(0, 0, 0, 0)
        project_layout.addWidget(open_btn)
        project_layout.addWidget(self.project_tree)
        project_tree_with_button.setLayout(project_layout)

        bottom_splitter = QSplitter(Qt.Horizontal)
        bottom_splitter.addWidget(project_tree_with_button)
        bottom_splitter.addWidget(summary_panel)
        bottom_splitter.addWidget(self.structure_tree)
        bottom_splitter.setSizes([300, 600, 300])


        left_splitter = QSplitter(Qt.Vertical)
        left_splitter.addWidget(self.code_viewer)
        left_splitter.addWidget(bottom_splitter)
        left_splitter.setSizes([300, 400])

        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.addWidget(left_splitter)
        main_splitter.addWidget(chat_widget)
        main_splitter.setSizes([1000, 400])

        container = QWidget()
        layout = QHBoxLayout()
        layout.addWidget(main_splitter)
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.pending_file_tasks = set()  


    def _open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Project")
        self.project_path = folder
        if not folder:
            return
        self.project_tree.clear()
        self._add_project_items(self.project_tree.invisibleRootItem(), folder)

        self.chat_input.setEnabled(False)
        self.send_btn.setEnabled(False)
        self.chat_input.setPlaceholderText(self.chat_placeholder_waiting)

        def enqueue_tasks():
            readme_path = os.path.join(folder, "README.md")
            if os.path.exists(readme_path):
                with open(readme_path, "r", encoding="utf-8") as f:
                    content = f.read()
                self.file_cache[readme_path] = {"desc": None, "source": content, "lines": [], "structure": []}
                self.task_queue.append(('readme', content, self.file_cache[readme_path]))

            for root, _, files in os.walk(folder):
                for file in files:
                    if file.endswith(".py"):
                        path = os.path.join(root, file)
                        self._cache_structure(path)
                        cache = self.file_cache.get(path)
                        if cache:
                            self.task_queue.append(('file', cache['source'], cache))
                            self.pending_file_tasks.add(path)

            self.total_tasks = len(self.task_queue)
            self.completed_tasks = 0
            self.progress_bar.setValue(0)

        def process_next():
            if not self.task_queue:
                self.chat_input.setEnabled(True)
                self.send_btn.setEnabled(True)
                return
            type, snippet, cache = self.task_queue.popleft()
            def on_finished(desc):
                self.completed_tasks += 1
                progress = int((self.completed_tasks / max(1, self.total_tasks)) * 100)
                self.progress_bar.setValue(progress)

                if type == 'readme' and self.last_clicked is None:
                    self._set_summary_viewer(desc)
                elif type == 'file':
                    for path, c in self.file_cache.items():
                        if c is cache:
                            self.pending_file_tasks.discard(path)
                    if not self.pending_file_tasks:
                        self.chat_input.setEnabled(True)
                        self.send_btn.setEnabled(True)
                        self.chat_input.setPlaceholderText(self.chat_placeholder_default)
                    for entry in cache.get('structure', []):
                        if entry['desc'] is None:
                            pass
                            # Uncomment to request summaries of functions and classes in the file when the project is opened.
                            #self.task_queue.append(('class' if entry['type'] == 'class' else 'func', entry['snippet'], entry))
                            #self.total_tasks += 1

                process_next()
            worker = LLMWorker(self.code_request.request, snippet, cache, type, on_finished)
            QThreadPool.globalInstance().start(worker)

        enqueue_tasks()
        process_next()

    def _contains_py(self, folder_path: str) -> bool:
        return any(file.endswith(".py") for _, _, files in os.walk(folder_path) for file in files)

    def _add_project_items(self, parent: QTreeWidgetItem, path: str):
        if not self._contains_py(path):
            return
        node = QTreeWidgetItem([os.path.basename(path)])
        node.setData(0, Qt.UserRole, path)
        parent.addChild(node)
        for entry in sorted(os.listdir(path)):
            full = os.path.join(path, entry)
            if os.path.isdir(full):
                self._add_project_items(node, full)
            elif entry.endswith(".py"):
                item = QTreeWidgetItem([entry])
                item.setData(0, Qt.UserRole, full)
                node.addChild(item)

    def _cache_structure(self, file_path: str):
        if file_path in self.file_cache:
            return
        try:
            with open(file_path, "r", encoding="utf-8") as fh:
                source = fh.read()
        except Exception as exc:
            self.file_cache[file_path] = {"error": str(exc)}
            return

        lines = source.splitlines()
        try:
            tree = ast.parse(source, filename=file_path)
        except Exception as exc:
            self.file_cache[file_path] = {"error": f"파싱 실패: {exc}", "source": source, "lines": lines}
            return

        structure = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                structure.append(self._node_info(node, lines, "function"))
            elif isinstance(node, ast.ClassDef):
                class_info = self._node_info(node, lines, "class")
                class_info["methods"] = [self._node_info(n, lines, "method")
                                           for n in node.body if isinstance(n, ast.FunctionDef)]
                structure.append(class_info)
        self.file_cache[file_path] = {"source": source, "lines": lines, "structure": structure, "desc": None}

    @staticmethod
    def _node_info(node: ast.AST, lines: list[str], kind: str):
        start = node.lineno
        end = getattr(node, "end_lineno", None)
        if end is None:
            end = start
            while end < len(lines) and (lines[end].startswith(" ") or lines[end].strip() == ""):
                end += 1
        snippet = "\n".join(lines[start - 1:end])
        return {"type": kind, "name": getattr(node, "name", "<anon>"), "lineno": start, "end_lineno": end, "snippet": snippet, "desc": None}

    def _on_project_item_clicked(self, item: QTreeWidgetItem):
        file_path = item.data(0, Qt.UserRole)
        if file_path == self.project_path:
            readme_path = os.path.join(file_path, "README.md")
            try:
                current_cache = self.file_cache[readme_path]
            except:
                current_cache = None
            if current_cache is not None and current_cache['desc'] is not None:
                self._set_summary_viewer(current_cache['desc'])
            else:
                self._set_summary_viewer("*Loading...*")
            self.code_viewer.clear()
            self.structure_tree.clear()
            self.last_clicked = None
            return
        if not (file_path and file_path.endswith(".py") and os.path.isfile(file_path)):
            return
        self.last_clicked = file_path
        self._cache_structure(file_path)
        info = self.file_cache.get(file_path, {})
        if "error" in info:
            self.code_viewer.setPlainText(info.get("error", "알 수 없는 에러"))
            self.summary_viewer.clear()
            self.structure_tree.clear()
            return

        self.code_viewer.setPlainText(info["source"])
        self._populate_structure_tree(info["structure"])
        self.summary_viewer.clear()
        self.current_file = file_path

        current_cache = self.file_cache[self.current_file]
        self.summary_viewer.setHtml("<i>Loading...</i>")
        self._set_summary_viewer("*Loading...*")
        if current_cache is not None and current_cache['desc'] is not None:
            self._set_summary_viewer(current_cache['desc'])
        else:
            self._set_summary_viewer("*Loading...*")

    def _populate_structure_tree(self, structure: list[dict]):
        self.structure_tree.clear()
        for entry in structure:
            if entry["type"] == "class":
                class_item = QTreeWidgetItem([f"Class: {entry['name']}"])
                class_item.setData(0, Qt.UserRole, entry)
                self.structure_tree.addTopLevelItem(class_item)
                for m in entry.get("methods", []):
                    m_item = QTreeWidgetItem([f"Method: {m['name']}"])
                    m_item.setData(0, Qt.UserRole, m)
                    class_item.addChild(m_item)
            else:
                func_item = QTreeWidgetItem([f"Function: {entry['name']}"])
                func_item.setData(0, Qt.UserRole, entry)
                self.structure_tree.addTopLevelItem(func_item)
        self.structure_tree.expandAll()

    def _on_structure_item_clicked(self, item: QTreeWidgetItem):
        info = item.data(0, Qt.UserRole)
        if not info:
            return

        lineno = info["lineno"]
        cursor = self.code_viewer.textCursor()
        block = self.code_viewer.document().findBlockByLineNumber(lineno - 1)
        cursor.setPosition(block.position())
        self.code_viewer.setTextCursor(cursor)

        scroll_bar = self.code_viewer.verticalScrollBar()
        scroll_bar.setValue(scroll_bar.maximum() * (lineno - 1) // self.code_viewer.blockCount())

        if info['type'] == 'class':
            current_type = 'class'
        elif info['type'] == 'method' or info['type'] == 'function':
            current_type = 'func'

        current_cache = None
        for entry in self.file_cache[self.current_file]["structure"]:
            if info["type"] == "class" or info["type"] == "function":
                if entry['type'] == info['type'] and entry['name'] == info['name'] and entry['lineno'] == info['lineno']:
                    current_cache = entry
                    break
            else:
                for method in entry.get("methods", []):
                    if method["name"] == info["name"] and method["lineno"] == info["lineno"]:
                        current_cache = method

        if info['type'] == "method" or "class":
            self._set_summary_viewer('*Loading...*')
            item_ref = weakref.ref(item)

            if current_cache is not None and current_cache['desc'] is not None:
                self._set_summary_viewer(current_cache['desc'])
                return

            def on_finished(desc):
                current_item = item_ref()
                if current_item is None:
                    return
                if self.structure_tree.currentItem() == current_item:
                    self._set_summary_viewer(desc)

            worker = LLMWorker(self.code_request.request, info['snippet'], current_cache, current_type, on_finished)
            QThreadPool.globalInstance().start(worker)
        else:
            if current_cache is not None and current_cache['desc'] is not None:
                self._set_summary_viewer(current_cache["desc"])
            else:
                self._set_summary_viewer("*Loading...*")

    def _set_summary_viewer(self, md):
        desc = md.replace('-', ' -').strip()
        html_body = markdown.markdown(desc, extensions=["fenced_code", "codehilite"], output_format="html5")
        style = HtmlFormatter(style="friendly").get_style_defs('.codehilite')
        full_html = f"<html><head><style>{style}</style></head><body>{html_body}</body></html>"
        self.summary_viewer.setHtml(full_html)
    
    def _send_chat_message(self):
        if not self.send_btn.isEnabled():
            return 

        message = self.chat_input.toPlainText().strip()
        if not message:
            return

        user_html = markdown.markdown(message, extensions=["fenced_code", "codehilite"], output_format="html5")
        self.chat_history_html += f"""
        <div class="user">
            <div class="user-msg">{user_html}</div>
        </div>
        """
        self._update_chat_output()
        self.chat_input.clear()

        def on_finished(response_text):
            response_text = response_text.replace('-', ' -').strip()
            ai_html = markdown.markdown(response_text, extensions=["fenced_code", "codehilite"], output_format="html5")
            self.chat_history_html += f"""
            <div>{ai_html}</div>
            """
            self._update_chat_output()

        summary = {
            file_name[len(self.project_path):]: self.file_cache[file_name]
            for file_name in self.file_cache
        }

        worker = ChatWorker(
            self.chat_request.request,
            inst=message,
            summary=summary,
            type="chat",
            callback=on_finished
        )
        QThreadPool.globalInstance().start(worker)


    def _update_chat_output(self):
        style = HtmlFormatter(style="friendly").get_style_defs('.codehilite')
        full_html = f"<html><head><style>{style}</style></head><body>{self.chat_history_html}</div></body></html>"
        self.chat_output.setHtml(full_html)
        self.chat_output.moveCursor(self.chat_output.textCursor().End)


if __name__ == "__main__":
    import traceback
    try:
        app = QApplication(sys.argv)
        window = FileBrowser()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        print("오류 발생:", e)
        traceback.print_exc()