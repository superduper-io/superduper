

def missing_ids(table1, table2):
    joined = table1.left_join(table2, table1.id == table2.id).projection(
        [table1, table2.id.name('table2_id')]
    )
    return joined.filter(joined.table2.id.isnull()).execute()
