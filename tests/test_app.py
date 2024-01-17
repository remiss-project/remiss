import os
import time

os.environ['PATH'] = '/run/host/usr/bin/:' + os.environ['PATH']
os.environ['LD_LIBRARY_PATH'] = '/run/host/usr/lib64/:/run/host/usr/lib64/samba/'
os.environ['REMISS_DEBUG'] = 'True'

from app import create_app


def test_001_click_wordcloud_sets_hashtag(dash_duo):
    app = create_app()
    dash_duo.start_server(app)
    dash_duo.wait_for_element("#placeholder-dashboard", timeout=4)
    dash_duo.multiple_click('#wordcloud-control', 1, delay=4)
    hashtags = dash_duo.get_session_storage('current-hashtags-state')
    assert hashtags == ['26M']
    assert dash_duo.get_logs() == [], "browser console should contain no error"


def test_002_click_wordcloud_changes_timeseries_plot(dash_duo):
    app = create_app()
    dash_duo.start_server(app)
    dash_duo.wait_for_element("#fig-tweet-ts", timeout=4)
    tweet_plot = dash_duo.find_element("#fig-tweet-ts")
    users_plot = dash_duo.find_element("#fig-users-ts")
    original_tweet_plot = tweet_plot.screenshot_as_base64
    original_user_plot = users_plot.screenshot_as_base64
    dash_duo.multiple_click('#wordcloud-control', 1, delay=1)
    dash_duo.wait_for_element("#fig-tweet-ts", timeout=4)
    tweet_plot = dash_duo.find_element("#fig-tweet-ts")
    users_plot = dash_duo.find_element("#fig-users-ts")
    expected_tweet_plot = tweet_plot.screenshot_as_base64
    expected_users_plot = users_plot.screenshot_as_base64
    assert original_tweet_plot != expected_tweet_plot, "tweet plot should change"
    assert original_user_plot != expected_users_plot, "users plot should change"
    assert dash_duo.get_logs() == [], "browser console should contain no error"


def test_003_date_range_changes_timeseries_plot(dash_duo):
    app = create_app()
    dash_duo.start_server(app)
    dash_duo.driver.maximize_window()
    dash_duo.wait_for_element("#fig-tweet-ts", timeout=4)
    before_tweet_plot_text = dash_duo.find_element("#fig-tweet-ts").text
    before_users_plot_text = dash_duo.find_element("#fig-users-ts").text
    date_picker_inputs = dash_duo.find_elements('.DateInput_input')
    date_picker_inputs = {x.accessible_name: x for x in date_picker_inputs}
    dash_duo.wait_for_element('.DateInput_input', timeout=4)
    date_picker_inputs['Start Date'].click()
    dash_duo.wait_for_element('.CalendarDay ', timeout=4)
    days = dash_duo.find_elements('.CalendarDay')
    days[len(days) // 2].click()
    date_picker_inputs['End Date'].click()
    dash_duo.wait_for_element('.CalendarDay ', timeout=4)
    days = dash_duo.find_elements('.CalendarDay')
    days[len(days) // 2 + 5].click()
    dash_duo.wait_for_element("#fig-tweet-ts", timeout=4)
    after_tweet_plot_text = dash_duo.find_element("#fig-tweet-ts").text
    after_users_plot_text = dash_duo.find_element("#fig-users-ts").text

    assert before_tweet_plot_text != after_tweet_plot_text, "tweet plot should change"
    assert before_users_plot_text != after_users_plot_text, "users plot should change"
    assert dash_duo.get_logs() == [], "browser console should contain no error"


def test_004_change_tweet_plot_dataset_dropdown(dash_duo):
    app = create_app()
    dash_duo.start_server(app)
    dash_duo.driver.maximize_window()
    dash_duo.wait_for_element("#fig-tweet-ts", timeout=4)
    before_tweet_plot_screenshot = dash_duo.find_element("#fig-tweet-ts").text
    dash_duo.wait_for_element('#dataset-dropdown-control', timeout=4)
    dash_duo.select_dcc_dropdown('#dataset-dropdown-control', 'test')

    dash_duo.wait_for_element("#fig-tweet-ts", timeout=4)
    after_tweet_plot_screenshot = dash_duo.find_element("#fig-tweet-ts").text

    assert before_tweet_plot_screenshot != after_tweet_plot_screenshot, "tweet plot should change"
    assert dash_duo.get_logs() == [], "browser console should contain no error"


def test_005_change_users_plot_dataset_dropdown(dash_duo):
    app = create_app()
    dash_duo.start_server(app)
    dash_duo.driver.maximize_window()
    dash_duo.wait_for_element("#fig-users-ts", timeout=4)
    before_users_plot_screenshot = dash_duo.find_element("#fig-users-ts").text
    dash_duo.wait_for_element('#dataset-dropdown-control', timeout=4)
    dash_duo.select_dcc_dropdown('#dataset-dropdown-control', 'test')

    dash_duo.wait_for_element("#fig-users-ts", timeout=4)
    after_users_plot_screenshot = dash_duo.find_element("#fig-users-ts").text

    assert before_users_plot_screenshot != after_users_plot_screenshot, "users plot should change"
    assert dash_duo.get_logs() == [], "browser console should contain no error"


def test_005_change_egonet_plot_dataset_dropdown(dash_duo):
    app = create_app()
    dash_duo.start_server(app)
    dash_duo.driver.maximize_window()
    dash_duo.wait_for_element("#fig-egonet", timeout=4)
    before_egonet_plot_screenshot = dash_duo.find_element("#fig-egonet").text
    dash_duo.wait_for_element('#dataset-dropdown-control', timeout=4)
    dash_duo.select_dcc_dropdown('#dataset-dropdown-control', 'test')

    dash_duo.wait_for_element("#fig-egonet", timeout=4)
    after_egonet_plot_screenshot = dash_duo.find_element("#fig-egonet").text

    assert before_egonet_plot_screenshot != after_egonet_plot_screenshot, "egonet plot should change"
    assert dash_duo.get_logs() == [], "browser console should contain no error"


def test_006_change_tweet_plot_table_hashtag_selection(dash_duo):
    app = create_app()
    dash_duo.start_server(app)
    dash_duo.driver.maximize_window()
    dash_duo.wait_for_element("#fig-tweet-ts", timeout=4)
    before_tweet_plot_screenshot = dash_duo.find_element("#fig-tweet-ts").text
    dash_duo.wait_for_element('.dash-cell-value', timeout=4)
    cell = dash_duo.find_element('.dash-cell-value')
    dash_duo.driver.execute_script("arguments[0].scrollIntoView();", cell)
    time.sleep(1)

    table = dash_duo.find_element('#table-top')
    time.sleep(1)
    table.click()
    dash_duo.wait_for_element("#fig-tweet-ts", timeout=4)
    after_tweet_plot_screenshot = dash_duo.find_element("#fig-tweet-ts").text

    assert before_tweet_plot_screenshot != after_tweet_plot_screenshot, "tweet plot should change"
    assert dash_duo.get_logs() == [], "browser console should contain no error"


def test_007_change_users_plot_table_hashtag_selection(dash_duo):
    app = create_app()
    dash_duo.start_server(app)
    dash_duo.driver.maximize_window()
    dash_duo.wait_for_element("#fig-users-ts", timeout=4)
    before_users_plot_screenshot = dash_duo.find_element("#fig-users-ts").text
    dash_duo.wait_for_element('#table-top', timeout=4)
    table = dash_duo.find_element('#table-top')
    dash_duo.driver.execute_script("arguments[0].scrollIntoView();", table)
    time.sleep(1)

    table.click()
    dash_duo.wait_for_element("#fig-users-ts", timeout=4)
    time.sleep(1)

    after_users_plot_screenshot = dash_duo.find_element("#fig-users-ts").text

    assert before_users_plot_screenshot != after_users_plot_screenshot, "users plot should change"
    assert dash_duo.get_logs() == [], "browser console should contain no error"


def test_008_change_user_egonet_plot_table_hashtag_selection(dash_duo):
    app = create_app()
    dash_duo.start_server(app)
    dash_duo.driver.maximize_window()
    dash_duo.wait_for_element("#fig-egonet", timeout=4)
    before_egonet_plot_screenshot = dash_duo.find_element("#fig-egonet").text
    dash_duo.wait_for_element('.dash-cell-value', timeout=4)
    cell = dash_duo.find_element('.dash-cell-value')
    dash_duo.driver.execute_script("arguments[0].scrollIntoView();", cell)
    time.sleep(1)
    table = dash_duo.find_element('#table-top')
    time.sleep(1)
    table.click()

    dash_duo.wait_for_element("#fig-egonet", timeout=4)
    after_egonet_plot_screenshot = dash_duo.find_element("#fig-egonet").text

    assert before_egonet_plot_screenshot != after_egonet_plot_screenshot, "egonet plot should change"
    assert dash_duo.get_logs() == [], "browser console should contain no error"
