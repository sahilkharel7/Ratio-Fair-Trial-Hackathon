from ratio.outcomes import registry,summary

def test_claim_specific_denominator_and_small_pattern_samples():
    records=registry()['records']
    all_cases=summary(records)
    assert (all_cases['merits_decisions'],all_cases['violation_found'],all_cases['no_violation'],all_cases['inadmissible'])==(5,4,1,2)
    assert all_cases['observed_merits_rate']==.8 and all_cases['success_probability'] is None
    narrow=summary(records,pattern='burden_shift')
    assert narrow['merits_decisions']==1 and narrow['observed_merits_rate'] is None

def test_other_forums_articles_unknown_claims_and_dissents_do_not_change_rate():
    rows=registry()['records']
    extras=[{**rows[0],'forum':'Other body'}, {**rows[0],'standard':'ICCPR Article 14(1)'}, {**rows[0],'outcome':'not_examined'}]
    assert summary(rows+extras)['observed_merits_rate']==.8
    majority=next(r for r in rows if r['id']=='ccpr-2526-2015')
    assert majority['outcome']=='no_violation' and 'majority' in majority['citation']

def test_all_examples_have_official_source_and_explicit_coding_limits():
    for row in registry()['records']:
        assert row['source_url'].startswith(('https://docstore.ohchr.org/','https://tbinternet.ohchr.org/','https://juris.ohchr.org/'))
        assert row['coding_status']=='source_checked_draft'
        assert row['implementation']=='not_tracked' and row['prediction'] is None
