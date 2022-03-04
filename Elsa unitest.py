import unittest


class MyTestCase(unittest.TestCase):
    def test_version_number(self):
        import json, urllib.request, updater
        url = "https://api.github.com/repos/georgerahul24/viraver1.1/tags"
        result = json.load(urllib.request.urlopen(url))
        result = tuple(filter(lambda x: x['name'].split(".")[0].lower().startswith("v") is False, result))
        latest_version = max(int(i['name'].split(".")[-1]) for i in result)
        self.assertEqual(updater.current_version, latest_version)  # add assertion here

    def test_adding_and_deleting_user(self):
        from Magic import export_import
        self.assertEqual(bool(export_import.export(mode = 'j')), True)

    def test_user_reading(self):
        from Magic import file_database
        self.assertEqual(file_database.check_user_from_file('admin'), '1234')
        self.assertEqual(file_database.write_to_file(' ', ' '), -1)
        self.assertEqual(file_database.write_to_file('admin', '1234'), 1)

    def test_indexer(self):
        from Magic import indexer
        self.assertNotEqual(len(open(indexer.indexerfolderpth).read()), 0)

    def test_program_run(self):
        from Magic import program_run
        print("Please close the notepad if it opens up to continue the test")
        self.assertEqual(program_run.program_run('notepad'), None)

    def test_plugins(self):
        import plugin_loader
        self.assertNotEqual(len(plugin_loader.pluginslist), 0)


if __name__ == '__main__':
    unittest.main()
