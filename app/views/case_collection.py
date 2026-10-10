"""The matter collection is the entry point; references stay in their own corpus."""
from __future__ import annotations

import streamlit as st

from ratio.display import md_escape
from ratio.extraction.loader import CaseManifest, LoaderError
from ratio.library import LibraryStore
from ratio.workspace import seed_collection
from ratio_ui import session, style

library=LibraryStore(session.store())
st.title('Case collection')
st.caption('Choose a court, open the case record, and review presumption of innocence. The collection is stored in SQLite on this computer.')

if st.button('Load the demo case collection',key='load_demo'):
    with st.spinner('Loading the recorded case and focused examples'):
        session.load_demo()
        seed_collection(library)
    st.rerun()

entries=library.list()
courts=sorted({row['court'] for row in entries})
selected=st.selectbox('Court',['All courts',*courts],key='library-court')
query=st.text_input('Find a case',placeholder='Defendant, charge or reference',key='library-search').strip().casefold()
shown=[e for e in entries if (selected=='All courts' or e['court']==selected) and
       (not query or query in f"{e['title']} {e['defendant']} {e['charge']} {e['case_id']}".casefold())]
st.caption(f'{len(shown)} cases · focused on ICCPR Article 14(2)')
if not shown:
    st.info('No case matches this selection. Import a record below or load the synthetic collection.')
for entry in shown:
    with style.panel('matter-'+entry['case_id']):
        title,action=st.columns([5,1],vertical_alignment='center')
        title.subheader(md_escape(entry['defendant']))
        title.caption(md_escape(f"{entry['court']} · {entry['case_id']} · {'Synthetic example' if entry['synthetic'] else 'Public record'}"))
        st.markdown(f"**Charge:** {md_escape(entry['charge'])}  \n**Prosecution seeks:** {md_escape(entry['penalty_summaries']['requested']['label'])}")
        prompts=entry['focus']['prompts']
        st.caption(f"{entry['document_count']} documents · {len(prompts)} source prompts to check. No prompt is a confirmed violation.")
        if action.button('Open case',key='open-'+entry['case_id'],type='primary'):
            session.open_case(entry['case_id']);st.switch_page('views/focus.py')

with st.expander('Add documents or assign staged PDFs to a case'):
    st.caption('Drop several PDFs, TXT or MD documents. Assign the documents to one case and confirm their types; a bulk classifier can use the same manifest contract.')
    provenance=st.selectbox('Material',['public','synthetic'],key='intake-provenance')
    source=st.text_input('Publication source (required for public material)',key='intake-source')
    confirmed=st.checkbox('These are public or synthetic documents and contain no confidential monitoring material.',key='intake-declaration')
    files=st.file_uploader('Documents',type=['pdf','txt','md'],accept_multiple_files=True,key='staged-upload')
    if st.button('Store documents for assignment',disabled=not(files and confirmed),key='stage-documents'):
        try:
            if len({f.name for f in files})!=len(files): raise LoaderError('Give files distinct names or upload them separately.')
            library.stage({f.name:f.getvalue() for f in files},provenance=provenance,source_note=source)
        except LoaderError as exc: st.error(md_escape(str(exc)))
        else: st.rerun()
    pending=library.pending_files()
    if pending:
        labels={f['id']:f['name'] for f in pending}
        picked=st.multiselect('Documents belonging to this case',list(labels),format_func=labels.get,key='assign-files')
        case_id=st.text_input('Case reference',placeholder='court-2025-001',key='assign-id')
        defendant=st.text_input('Defendant',key='assign-defendant')
        case_title=st.text_input('Case title (as written in the record)',key='assign-title')
        court=st.text_input('Court and chamber',value=selected if selected!='All courts' else '',key='assign-court')
        charge=st.text_input('Charge (leave blank if it needs verification)',key='assign-charge')
        grouped=[f for f in pending if f['id'] in picked]
        types={}
        for file in grouped:
            if file['problem']: st.warning(md_escape(f"{file['name']}: {file['problem']}"))
            types[file['id']]=st.selectbox(f"Document type: {file['name']}",['indictment','judgment','monitoring_note','transcript','detention_order'],key='type-'+file['id'])
        grouped_confirmed=st.checkbox('The selected documents belong to this one case and their document types are correct.',key='assign-declaration')
        if st.button('Create case in SQLite',type='primary',disabled=not(grouped and grouped_confirmed),key='create-matter'):
            try:
                origin=grouped[0]['provenance']
                manifest=CaseManifest(case_id=case_id,title=case_title or defendant,court=court,charge_type=charge or 'Not recorded',
                                      data_provenance=origin,synthetic=origin=='synthetic',source_note=grouped[0]['source_note'] or None,
                                      documents=[{'path':f['name'],'title':f['name'],'type':types[f['id']]} for f in grouped])
                library.import_manifest(manifest,{f['name']:f['id'] for f in grouped},defendant=defendant)
                session.open_case(case_id);st.switch_page('views/focus.py')
            except (LoaderError,ValueError) as exc: st.error(md_escape(str(exc)))
