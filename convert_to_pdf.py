#!/usr/bin/env python3
"""
Convert Markdown file to PDF with Mermaid diagram support
"""
import markdown
from weasyprint import HTML, CSS
from pathlib import Path
import re
import base64

def convert_mermaid_to_reference(md_content):
    """
    Convert Mermaid diagrams to a reference note pointing to the flowchart image file.
    """
    # Find mermaid code blocks
    mermaid_pattern = r'```mermaid\n(.*?)```'

    def replace_mermaid(match):
        # Add a reference to the flowchart image
        return '''
<div style="border: 3px solid #0066cc; padding: 25px; margin: 30px 0; background-color: #f0f8ff; border-radius: 8px; page-break-inside: avoid;">
    <h3 style="color: #0066cc; margin-top: 0; font-size: 1.3em;">📊 Strategy Build Process Flowchart</h3>
    <p style="font-size: 1.1em; margin: 15px 0;"><strong>The complete flowchart diagram is available in the following files:</strong></p>
    <ul style="font-size: 1.05em; line-height: 1.8;">
        <li><strong>PNG Image:</strong> <code>docs/flowchart.png</code> - High-resolution flowchart image</li>
        <li><strong>Mermaid Source:</strong> <code>docs/flowchart.mmd</code> - Editable Mermaid diagram source</li>
        <li><strong>Markdown File:</strong> <code>docs/STRATEGY_BUILD_FLOWCHART.md</code> - Interactive version (view in VS Code, GitHub, or any Mermaid-compatible viewer)</li>
    </ul>
    <div style="background-color: #e6f3ff; padding: 15px; border-radius: 5px; margin-top: 20px;">
        <p style="margin: 0; font-size: 1.05em;"><strong>💡 Tip:</strong> Open <code>flowchart.png</code> to see the complete color-coded flowchart showing:</p>
        <ul style="margin: 10px 0 0 20px; font-size: 1.05em;">
            <li>🔵 <strong>Blue boxes</strong> - LLM operations (Claude Opus 4.6 with Extended Thinking)</li>
            <li>🟠 <strong>Orange boxes</strong> - ML training and backtesting steps</li>
            <li>🟣 <strong>Purple boxes</strong> - Scoring and evaluation</li>
            <li>🟢 <strong>Green boxes</strong> - Final iteration selection</li>
        </ul>
    </div>
    <p style="margin: 20px 0 0 0; font-size: 0.95em; color: #666;"><em>The flowchart visualizes the complete algorithm build process from initial design through iteration loops to final strategy deployment and retraining.</em></p>
</div>
'''

    return re.sub(mermaid_pattern, replace_mermaid, md_content, flags=re.DOTALL)

def main():
    # Read the markdown file
    md_file = Path('docs/STRATEGY_BUILD_FLOWCHART.md')
    pdf_file = Path('docs/STRATEGY_BUILD_FLOWCHART.pdf')

    print(f"Reading {md_file}...")
    md_content = md_file.read_text()

    # Convert Mermaid diagrams to reference notes
    print("Converting Mermaid diagram to reference note...")
    md_content = convert_mermaid_to_reference(md_content)
    
    # Convert markdown to HTML
    print("Converting Markdown to HTML...")
    html_content = markdown.markdown(md_content, extensions=['extra', 'codehilite', 'tables'])
    
    # Wrap in HTML document with styling
    full_html = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Strategy Build Flowchart</title>
        <style>
            @page {{
                size: A4;
                margin: 2cm;
            }}
            body {{
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 100%;
            }}
            h1 {{
                color: #0066cc;
                border-bottom: 3px solid #0066cc;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #0066cc;
                border-bottom: 2px solid #e0e0e0;
                padding-bottom: 8px;
                margin-top: 30px;
            }}
            h3 {{
                color: #333;
                margin-top: 20px;
            }}
            code {{
                background-color: #f5f5f5;
                padding: 2px 6px;
                border-radius: 3px;
                font-family: "Monaco", "Courier New", monospace;
                font-size: 0.9em;
            }}
            pre {{
                background-color: #f5f5f5;
                padding: 15px;
                border-radius: 5px;
                overflow-x: auto;
                border-left: 4px solid #0066cc;
            }}
            ul, ol {{
                margin-left: 20px;
            }}
            li {{
                margin-bottom: 5px;
            }}
            strong {{
                color: #000;
            }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    '''
    
    # Convert HTML to PDF
    print(f"Generating PDF at {pdf_file}...")
    HTML(string=full_html).write_pdf(pdf_file)
    
    print(f"✅ Successfully created {pdf_file}")
    print(f"📄 File size: {pdf_file.stat().st_size / 1024:.1f} KB")

if __name__ == '__main__':
    main()

